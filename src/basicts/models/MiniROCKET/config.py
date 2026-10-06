from dataclasses import dataclass, field
from typing import Sequence

from basicts.configs import BasicTSModelConfig


@dataclass
class MiniROCKETConfig(BasicTSModelConfig):
    input_len: int = field(default=None)
    num_features: int = field(default=None)
    num_classes: int = field(default=None)
    num_kernels: int = field(default=1024)
    kernel_sizes: Sequence[int] = field(default=(7, 9, 11))
    dilations: Sequence[int] = field(default=(1, 2, 4, 8, 16))
    seed: int = field(default=42)
    dropout: float = field(default=0.1)
