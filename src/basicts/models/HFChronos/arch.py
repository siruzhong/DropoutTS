import inspect
import sys
import types
from contextlib import nullcontext
from importlib.machinery import ModuleSpec
from typing import Optional

import torch
from torch import nn

from .config import HFChronosAdapterConfig


def _ensure_sklearn_stub():
    try:
        import sklearn.preprocessing  # noqa: F401
        return
    except ImportError:
        pass

    class _IdentityEncoder:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, values, y=None):
            return self

        def fit_transform(self, values, y=None):
            return values

        def transform(self, values):
            return values

    sklearn_module = types.ModuleType("sklearn")
    preprocessing_module = types.ModuleType("sklearn.preprocessing")
    metrics_module = types.ModuleType("sklearn.metrics")
    sklearn_module.__spec__ = ModuleSpec("sklearn", loader=None)
    preprocessing_module.__spec__ = ModuleSpec("sklearn.preprocessing", loader=None)
    metrics_module.__spec__ = ModuleSpec("sklearn.metrics", loader=None)
    sklearn_module.__path__ = []
    preprocessing_module.OrdinalEncoder = _IdentityEncoder
    preprocessing_module.TargetEncoder = _IdentityEncoder
    metrics_module.roc_curve = lambda *args, **kwargs: ([], [], [])
    sklearn_module.preprocessing = preprocessing_module
    sklearn_module.metrics = metrics_module
    sys.modules.setdefault("sklearn", sklearn_module)
    sys.modules.setdefault("sklearn.preprocessing", preprocessing_module)
    sys.modules.setdefault("sklearn.metrics", metrics_module)


def _patch_t5stack_signature():
    try:
        from transformers.models.t5 import modeling_t5
    except ImportError:
        return
    signature = inspect.signature(modeling_t5.T5Stack.__init__)
    if len(signature.parameters) > 2:
        return

    original_cls = modeling_t5.T5Stack

    class _CompatT5Stack(original_cls):
        def __init__(self, config, embed_tokens=None):
            super().__init__(config)
            if embed_tokens is not None:
                self.embed_tokens = embed_tokens

    modeling_t5.T5Stack = _CompatT5Stack


def _patch_chronos_classes():
    try:
        from chronos.chronos_bolt import ChronosBoltModelForForecasting
    except ImportError:
        return
    if isinstance(getattr(ChronosBoltModelForForecasting, "_tied_weights_keys", None), list):
        ChronosBoltModelForForecasting._tied_weights_keys = {
            "encoder.embed_tokens.weight": "shared.weight",
            "decoder.embed_tokens.weight": "shared.weight",
        }


def _import_chronos(source_dir: Optional[str]):
    if source_dir:
        sys.path.insert(0, source_dir)
    _ensure_sklearn_stub()
    _patch_t5stack_signature()
    try:
        from chronos import BaseChronosPipeline
        _patch_chronos_classes()
        return BaseChronosPipeline
    except ImportError:
        try:
            from chronos.chronos2.pipeline import Chronos2Pipeline
            _patch_chronos_classes()
            return Chronos2Pipeline
        except ImportError as err:
            raise ImportError(
                "Chronos is not importable. Install chronos-forecasting or pass "
                "HFChronosAdapterConfig(source_dir='<chronos-forecasting>/src')."
            ) from err


class HFChronosAdapterForForecasting(nn.Module):
    """Frozen Chronos/Chronos-2 backbone with a trainable residual adapter."""

    def __init__(self, config: HFChronosAdapterConfig):
        super().__init__()
        self.input_len = config.input_len
        self.output_len = config.output_len
        self.num_features = config.num_features
        self.freeze_backbone = config.freeze_backbone
        self.use_adapter = config.use_adapter
        self.batch_size = config.batch_size
        self.context_length = config.context_length or config.input_len

        pipeline_cls = _import_chronos(config.source_dir)
        self.pipeline = pipeline_cls.from_pretrained(
            config.model_name_or_path,
            local_files_only=config.local_files_only,
        )
        if hasattr(self.pipeline, "model") and isinstance(self.pipeline.model, nn.Module):
            self.backbone = self.pipeline.model
            if self.freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
        else:
            self.backbone = None

        self.adapter = None
        if self.use_adapter:
            self.adapter = nn.Sequential(
                nn.LayerNorm(config.num_features),
                nn.Linear(config.num_features, config.adapter_hidden_size),
                nn.GELU(),
                nn.Dropout(config.adapter_dropout),
                nn.Linear(config.adapter_hidden_size, config.num_features),
            )

    def _predict(self, context: torch.Tensor) -> torch.Tensor:
        series = [item.contiguous() for item in context]
        with torch.no_grad() if self.freeze_backbone else nullcontext():
            try:
                outputs = self.pipeline.predict(
                    series,
                    prediction_length=self.output_len,
                    batch_size=self.batch_size,
                    context_length=self.context_length,
                    limit_prediction_length=False,
                )
            except TypeError:
                outputs = self.pipeline.predict(
                    series,
                    prediction_length=self.output_len,
                    limit_prediction_length=False,
                )
        if isinstance(outputs, list):
            outputs = torch.stack([
                item if item.ndim == 1 else item.median(dim=0).values
                for item in outputs
            ])
        if outputs.ndim == 3:
            outputs = outputs.median(dim=1).values
        return outputs.to(context.device, dtype=context.dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("Chronos adapter expects [batch, input_len, num_features] inputs.")
        batch_size, _, num_features = inputs.shape
        if num_features != self.num_features:
            raise ValueError(f"Expected num_features={self.num_features}, got {num_features}.")
        context = inputs[:, -self.context_length:, :].transpose(1, 2).reshape(-1, self.context_length)
        prediction = self._predict(context)
        prediction = prediction.reshape(batch_size, num_features, -1).transpose(1, 2)
        prediction = prediction[:, :self.output_len, :]
        return prediction if self.adapter is None else prediction + self.adapter(prediction)
