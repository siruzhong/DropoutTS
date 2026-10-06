from contextlib import nullcontext
import sys
import os
from typing import List, Optional

import numpy as np
import torch
from torch import nn

from .config import HFTimesFMAdapterConfig


def _load_timesfm_state_dict(model, weights_path: str) -> None:
    module = model.model if hasattr(model, "model") else model
    try:
        module.load_checkpoint(weights_path, torch_compile=False)
        return
    except TypeError:
        pass
    except RuntimeError as err:
        _load_timesfm_state_dict_with_qkv_compat(module, weights_path, err)
        return

    try:
        if hasattr(model, "load_checkpoint"):
            model.load_checkpoint(path=weights_path)
        else:
            module.load_checkpoint(weights_path)
    except RuntimeError as err:
        _load_timesfm_state_dict_with_qkv_compat(module, weights_path, err)


def _load_timesfm_state_dict_with_qkv_compat(module, weights_path: str, original_err: RuntimeError) -> None:
    try:
        from safetensors.torch import load_file

        tensors = dict(load_file(weights_path))
        expected = module.state_dict()
        for key in list(tensors):
            if key.endswith(".attn.qkv_proj.weight"):
                prefix = key[:-len("qkv_proj.weight")]
                query_key = prefix + "query.weight"
                key_key = prefix + "key.weight"
                value_key = prefix + "value.weight"
                if all(name in expected for name in (query_key, key_key, value_key)):
                    tensors[query_key], tensors[key_key], tensors[value_key] = tensors[key].chunk(3, dim=0)
                    if key not in expected:
                        del tensors[key]
            elif key.endswith(".attn.qkv_proj.bias"):
                prefix = key[:-len("qkv_proj.bias")]
                query_key = prefix + "query.bias"
                key_key = prefix + "key.bias"
                value_key = prefix + "value.bias"
                if all(name in expected for name in (query_key, key_key, value_key)):
                    tensors[query_key], tensors[key_key], tensors[value_key] = tensors[key].chunk(3, dim=0)
                    if key not in expected:
                        del tensors[key]
        module.load_state_dict(tensors, strict=True)
        if hasattr(module, "device"):
            module.to(module.device)
        module.eval()
    except Exception as compat_err:
        raise original_err from compat_err


def _load_native_timesfm(config: HFTimesFMAdapterConfig):
    if config.source_dir:
        sys.path.insert(0, config.source_dir)
    try:
        from timesfm import TimesFM_2p5_200M_torch
        try:
            from timesfm import ForecastConfig
        except ImportError:
            from timesfm.configs import ForecastConfig
    except ImportError as err:
        raise ImportError(
            "TimesFM native PyTorch backend is not importable. Pass "
            "HFTimesFMAdapterConfig(source_dir='<timesfm>/src') or install timesfm."
        ) from err

    try:
        model = TimesFM_2p5_200M_torch(torch_compile=False)
    except TypeError:
        model = TimesFM_2p5_200M_torch()
    if os.path.isdir(config.model_name_or_path):
        weights_path = os.path.join(config.model_name_or_path, "model.safetensors")
    else:
        from huggingface_hub import hf_hub_download
        weights_path = hf_hub_download(
            repo_id=config.model_name_or_path,
            filename="model.safetensors",
            cache_dir=config.cache_dir,
            local_files_only=config.local_files_only,
        )
    _load_timesfm_state_dict(model, weights_path)
    model.compile(ForecastConfig(
        max_context=config.forecast_context_len or config.input_len,
        max_horizon=config.output_len,
        per_core_batch_size=config.per_core_batch_size,
        normalize_inputs=False,
        use_continuous_quantile_head=False,
    ))
    return model


def _load_transformers_timesfm(config: HFTimesFMAdapterConfig):
    try:
        from transformers import AutoModelForTimeSeriesPrediction
    except ImportError as err:
        raise ImportError(
            "This transformers version does not provide AutoModelForTimeSeriesPrediction. "
            "Use the native TimesFM PyTorch checkpoint/backend instead."
        ) from err

    return AutoModelForTimeSeriesPrediction.from_pretrained(
        config.model_name_or_path,
        cache_dir=config.cache_dir,
        local_files_only=config.local_files_only,
    )


class HFTimesFMAdapterForForecasting(nn.Module):
    """Frozen HuggingFace TimesFM backbone with a trainable residual adapter."""

    def __init__(self, config: HFTimesFMAdapterConfig):
        super().__init__()
        self.input_len = config.input_len
        self.output_len = config.output_len
        self.num_features = config.num_features
        self.freeze_backbone = config.freeze_backbone
        self.use_adapter = config.use_adapter
        self.forecast_context_len = config.forecast_context_len or config.input_len
        self.backend = "native"

        try:
            self.backbone = _load_native_timesfm(config)
        except ImportError:
            self.backend = "transformers"
            self.backbone = _load_transformers_timesfm(config)
        self._validate_backbone_shape(config)

        if self.freeze_backbone and isinstance(self.backbone, nn.Module):
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.adapter = None
        if self.use_adapter:
            self.adapter = nn.Sequential(
                nn.LayerNorm(config.num_features),
                nn.Linear(config.num_features, config.adapter_hidden_size),
                nn.GELU(),
                nn.Dropout(config.adapter_dropout),
                nn.Linear(config.adapter_hidden_size, config.num_features),
            )

    def _validate_backbone_shape(self, config: HFTimesFMAdapterConfig) -> None:
        if self.backend == "native":
            context_length = getattr(self.backbone.model.config, "context_limit", None)
            horizon_length = None
        else:
            hf_cfg = self.backbone.config
            context_length = getattr(hf_cfg, "context_length", None)
            horizon_length = getattr(hf_cfg, "horizon_length", None)
        mismatches = []
        if context_length is not None and config.input_len > context_length:
            mismatches.append(f"input_len={config.input_len} exceeds context_length={context_length}")
        if self.backend == "native" and context_length is not None and config.input_len + config.output_len > context_length:
            mismatches.append(
                f"input_len + output_len={config.input_len + config.output_len} exceeds context_limit={context_length}"
            )
        if horizon_length is not None and config.output_len > horizon_length:
            mismatches.append(f"output_len={config.output_len} exceeds horizon_length={horizon_length}")
        if mismatches:
            raise ValueError(
                "TimesFM checkpoint cannot satisfy the requested forecast shape: "
                + "; ".join(mismatches)
            )

    def _flatten_series(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        batch_size, seq_len, num_features = inputs.shape
        series = []
        for b in range(batch_size):
            for f in range(num_features):
                series.append(inputs[b, -self.forecast_context_len:, f].contiguous())
        return series

    def _native_forecast(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, _, num_features = inputs.shape
        model = self.backbone.model
        context = inputs[:, -self.forecast_context_len:, :].transpose(1, 2).reshape(-1, self.forecast_context_len)
        context = context.to(device=model.device, dtype=torch.float32)

        max_context = getattr(self.backbone.forecast_config, "max_context", self.forecast_context_len)
        if context.shape[-1] > max_context:
            context = context[:, -max_context:]
            masks = torch.zeros_like(context, dtype=torch.bool)
            pad_len = 0
        else:
            pad_len = max_context - context.shape[-1]
            if pad_len > 0:
                padding = torch.zeros(context.shape[0], pad_len, device=context.device, dtype=context.dtype)
                context = torch.cat([padding, context], dim=-1)
            masks = torch.zeros_like(context, dtype=torch.bool)
        if pad_len > 0:
            masks[:, :pad_len] = True

        def _decode_full(series: torch.Tensor) -> torch.Tensor:
            pf_outputs, _, ar_outputs = model.decode(self.output_len, series, masks)
            to_concat = [pf_outputs[:, -1, ...]]
            if ar_outputs is not None:
                to_concat.append(ar_outputs.reshape(series.shape[0], -1, model.q))
            return torch.cat(to_concat, dim=1)[:, :self.output_len, :]

        full_forecast = _decode_full(context)
        forecast_config = getattr(self.backbone, "forecast_config", None)
        if getattr(forecast_config, "force_flip_invariance", False):
            full_forecast = (full_forecast - _decode_full(-context)) / 2
        if getattr(forecast_config, "infer_is_positive", False):
            is_positive = torch.all(context >= 0, dim=-1, keepdim=True)
            full_forecast = torch.where(
                is_positive[..., None],
                torch.maximum(full_forecast, torch.zeros_like(full_forecast)),
                full_forecast,
            )

        point_forecast = full_forecast[..., model.aridx]
        prediction = point_forecast.to(device=inputs.device, dtype=inputs.dtype)
        return prediction.reshape(batch_size, num_features, -1).transpose(1, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("TimesFM adapter expects [batch, input_len, num_features] inputs.")
        if inputs.shape[1] != self.input_len:
            raise ValueError(f"Expected input_len={self.input_len}, got {inputs.shape[1]}.")
        if inputs.shape[2] != self.num_features:
            raise ValueError(f"Expected num_features={self.num_features}, got {inputs.shape[2]}.")

        backbone_ctx = torch.no_grad() if self.freeze_backbone else nullcontext()
        if self.backend == "native":
            pred = self._native_forecast(inputs)
            pred = pred[:, :self.output_len, :self.num_features]
            return pred if self.adapter is None else pred + self.adapter(pred)

        if self.freeze_backbone:
            self.backbone.eval()

        series = self._flatten_series(inputs)
        with backbone_ctx:
            outputs = self.backbone(past_values=series, forecast_context_len=self.forecast_context_len)
            prediction = outputs.mean_predictions

        if prediction is None:
            prediction = outputs.full_predictions[..., 0]

        batch_size = inputs.shape[0]
        pred = prediction.reshape(batch_size, self.num_features, -1).transpose(1, 2)
        pred = pred[:, :self.output_len, :self.num_features]
        return pred if self.adapter is None else pred + self.adapter(pred)
