from dataclasses import dataclass, field
from typing import Optional

from basicts.configs import BasicTSModelConfig


@dataclass
class HFTimesFMAdapterConfig(BasicTSModelConfig):
    """Config for frozen HuggingFace TimesFM lightweight adaptation."""

    model_name_or_path: str = field(
        default="checkpoints/tsfm/timesfm-2.5-200m-pytorch",
        metadata={"help": "Local TimesFM checkpoint path or HuggingFace repo id."})
    source_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Optional <timesfm>/src directory for the native PyTorch TimesFM backend."})
    cache_dir: Optional[str] = field(default=None)
    local_files_only: bool = field(default=True)

    input_len: int = field(default=512)
    output_len: int = field(default=96)
    num_features: int = field(default=7)

    freeze_backbone: bool = field(default=True)
    use_adapter: bool = field(default=True)
    adapter_hidden_size: int = field(default=128)
    adapter_dropout: float = field(default=0.1)
    forecast_context_len: Optional[int] = field(default=None)
    per_core_batch_size: int = field(default=32)
