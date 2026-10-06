from __future__ import annotations

from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class TimeFilterConfig(BasicTSModelConfig):
    """
    TimeFilter config adapted to `basicts`.

    Upstream constraints:
    - `input_len` must be divisible by `patch_len` (no padding in the reference implementation).
    - `d_model` must be divisible by `n_heads`.
    """

    # Required by BasicTSLauncher / run_baselines.py
    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length for forecasting task."})
    num_features: int = field(default=None, metadata={"help": "Number of features / variables."})

    # Upstream TimeFilter hyperparameters
    d_model: int = field(default=64, metadata={"help": "Model hidden dimension."})
    d_ff: int = field(default=256, metadata={"help": "FFN hidden dimension (GraphBlock d_ff)."})
    patch_len: int = field(default=8, metadata={"help": "Patch length (patch_len). Must divide input_len."})
    alpha: float = field(default=0.1, metadata={"help": "Graph sparsity coefficient used in mask_topk."})
    top_p: float = field(default=0.5, metadata={"help": "MoE-like routing threshold."})
    n_heads: int = field(default=4, metadata={"help": "Number of graph heads."})
    e_layers: int = field(default=3, metadata={"help": "Number of TimeFilter graph blocks."})
    dropout: float = field(default=0.1, metadata={"help": "Dropout probability."})

    pos: bool = field(default=True, metadata={"help": "Whether to add positional embedding in PatchEmbed."})

    def __post_init__(self) -> None:
        if self.input_len is None or self.output_len is None or self.num_features is None:
            # Let the runner populate these values later.
            return

        if self.patch_len <= 0:
            raise ValueError(f"patch_len must be positive, got {self.patch_len}")
        if self.input_len % self.patch_len != 0:
            raise ValueError(
                f"TimeFilter requires input_len to be divisible by patch_len. "
                f"Got input_len={self.input_len}, patch_len={self.patch_len}."
            )
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads. Got d_model={self.d_model}, n_heads={self.n_heads}.")


__all__ = ["TimeFilterConfig"]

