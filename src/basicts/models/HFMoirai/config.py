from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from basicts.configs import BasicTSModelConfig


@dataclass
class HFMoiraiAdapterConfig(BasicTSModelConfig):
    model_name_or_path: str = field(default="checkpoints/tsfm/moirai-1.1-R-small")
    source_dir: Optional[str] = field(default=None)
    model_family: Literal["moirai", "moirai_moe"] = field(default="moirai")

    input_len: int = field(default=96)
    output_len: int = field(default=96)
    num_features: int = field(default=7)

    freeze_backbone: bool = field(default=True)
    use_adapter: bool = field(default=True)
    adapter_hidden_size: int = field(default=128)
    adapter_dropout: float = field(default=0.1)
    patch_size: Union[int, str] = field(default=16)
    num_samples: int = field(default=10)
