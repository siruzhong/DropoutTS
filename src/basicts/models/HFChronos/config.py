from dataclasses import dataclass, field
from typing import Optional

from basicts.configs import BasicTSModelConfig


@dataclass
class HFChronosAdapterConfig(BasicTSModelConfig):
    model_name_or_path: str = field(default="checkpoints/tsfm/chronos-2")
    source_dir: Optional[str] = field(default=None)
    local_files_only: bool = field(default=True)

    input_len: int = field(default=96)
    output_len: int = field(default=96)
    num_features: int = field(default=7)

    freeze_backbone: bool = field(default=True)
    use_adapter: bool = field(default=True)
    adapter_hidden_size: int = field(default=128)
    adapter_dropout: float = field(default=0.1)
    batch_size: int = field(default=256)
    context_length: Optional[int] = field(default=None)
