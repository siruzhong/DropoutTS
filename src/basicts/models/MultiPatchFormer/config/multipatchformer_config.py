from __future__ import annotations

from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class MultiPatchFormerConfig(BasicTSModelConfig):
    """
    MultiPatchFormer configuration adapted from THUML Time-Series-Library.

    The reference model uses four fixed patch scales:
    (8, 8), (16, 8), (24, 7), and (32, 6), followed by temporal and
    variable-wise Transformer encoders.
    """

    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length."})
    num_features: int = field(default=None, metadata={"help": "Number of variables."})

    d_model: int = field(default=256, metadata={"help": "Model hidden dimension."})
    d_ff: int = field(default=1024, metadata={"help": "Feed-forward hidden dimension."})
    n_heads: int = field(default=8, metadata={"help": "Number of attention heads."})
    e_layers: int = field(default=3, metadata={"help": "Number of encoder layers per stage."})
    dropout: float = field(default=0.1, metadata={"help": "Dropout rate."})
    activation: str = field(default="gelu", metadata={"help": "Feed-forward activation."})
    output_attention: bool = field(default=False, metadata={"help": "Return attention weights."})

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model must be divisible by n_heads. Got d_model={self.d_model}, n_heads={self.n_heads}."
            )
        if self.input_len is not None and self.input_len < 32:
            raise ValueError(f"MultiPatchFormer requires input_len >= 32, got {self.input_len}.")


__all__ = ["MultiPatchFormerConfig"]
