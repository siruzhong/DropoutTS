import sys
from contextlib import nullcontext
from typing import Optional

import torch
from torch import nn

from .config import HFMoiraiAdapterConfig


class _PyTree:
    def __class_getitem__(cls, item):
        return cls


def _patch_jaxtyping():
    try:
        import jaxtyping
        if not hasattr(jaxtyping, "PyTree"):
            jaxtyping.PyTree = _PyTree
    except ImportError:
        return


def _import_moirai(config: HFMoiraiAdapterConfig):
    if config.source_dir:
        sys.path.insert(0, config.source_dir)
    _patch_jaxtyping()
    try:
        if config.model_family == "moirai_moe":
            from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
            return MoiraiMoEForecast, MoiraiMoEModule
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        return MoiraiForecast, MoiraiModule
    except ImportError as err:
        raise ImportError(
            "Uni2TS/Moirai is not importable. Install uni2ts or pass "
            "HFMoiraiAdapterConfig(source_dir='<uni2ts>/src')."
        ) from err


class HFMoiraiAdapterForForecasting(nn.Module):
    """Frozen Moirai/Moirai-MoE backbone with a trainable residual adapter."""

    def __init__(self, config: HFMoiraiAdapterConfig):
        super().__init__()
        self.input_len = config.input_len
        self.output_len = config.output_len
        self.num_features = config.num_features
        self.freeze_backbone = config.freeze_backbone
        self.use_adapter = config.use_adapter
        self.num_samples = config.num_samples

        forecast_cls, module_cls = _import_moirai(config)
        module = module_cls.from_pretrained(config.model_name_or_path)
        self.backbone = forecast_cls(
            module=module,
            prediction_length=config.output_len,
            context_length=config.input_len,
            patch_size=config.patch_size,
            num_samples=config.num_samples,
            target_dim=config.num_features,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        if self.freeze_backbone:
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("Moirai adapter expects [batch, input_len, num_features] inputs.")
        observed = torch.ones_like(inputs, dtype=torch.bool)
        is_pad = torch.zeros(inputs.shape[0], inputs.shape[1], device=inputs.device, dtype=torch.bool)
        backbone_ctx = torch.no_grad() if self.freeze_backbone else nullcontext()
        if self.freeze_backbone:
            self.backbone.eval()
        with backbone_ctx:
            prediction = self.backbone(
                past_target=inputs,
                past_observed_target=observed,
                past_is_pad=is_pad,
                num_samples=self.num_samples,
            )
        if prediction.ndim == 4:
            prediction = prediction.median(dim=1).values
        elif prediction.ndim == 3:
            if prediction.shape[-1] == self.num_features and prediction.shape[1] >= self.output_len:
                pass
            elif self.num_features == 1 and prediction.shape[-1] >= self.output_len:
                prediction = prediction.median(dim=1).values.unsqueeze(-1)
            else:
                raise ValueError(
                    "Moirai backbone returned an unsupported 3D shape. Expected "
                    "[batch, horizon, features] or [batch, samples, horizon], "
                    f"got {tuple(prediction.shape)}."
                )
        else:
            raise ValueError(
                "Moirai backbone should return [batch, horizon, features] or "
                "[batch, samples, horizon, features]."
            )
        prediction = prediction[:, :self.output_len, :self.num_features]
        if prediction.shape[1] != self.output_len or prediction.shape[2] != self.num_features:
            raise ValueError(
                "Moirai prediction shape does not match the requested forecast shape: "
                f"got {tuple(prediction.shape)}, expected "
                f"({inputs.shape[0]}, {self.output_len}, {self.num_features})."
            )
        return prediction if self.adapter is None else prediction + self.adapter(prediction)
