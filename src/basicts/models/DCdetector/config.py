from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class DCdetectorConfig(BasicTSModelConfig):
    input_len: int = field(default=None)
    output_len: int = field(default=None)
    num_features: int = field(default=None)
    hidden_size: int = field(default=128)
    num_layers: int = field(default=2)
    num_heads: int = field(default=4)
    intermediate_size: int = field(default=256)
    dropout: float = field(default=0.1)
    branch_agreement_weight: float = field(default=0.1)
