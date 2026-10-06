from dataclasses import dataclass, field
from typing import Sequence

from basicts.configs import BasicTSModelConfig


@dataclass
class InceptionTimeConfig(BasicTSModelConfig):
    input_len: int = field(default=None)
    num_features: int = field(default=None)
    num_classes: int = field(default=None)
    hidden_size: int = field(default=32)
    bottleneck_size: int = field(default=32)
    kernel_sizes: Sequence[int] = field(default=(9, 19, 39))
    num_blocks: int = field(default=6)
    dropout: float = field(default=0.1)
