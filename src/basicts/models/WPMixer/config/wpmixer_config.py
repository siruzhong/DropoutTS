from __future__ import annotations

from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class WPMixerConfig(BasicTSModelConfig):
    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length."})
    num_features: int = field(default=None, metadata={"help": "Number of features."})

    # Backbone hyperparameters
    d_model: int = field(default=64, metadata={"help": "Mixer embedding dimension."})
    dropout: float = field(default=0.1, metadata={"help": "Dropout rate."})
    tfactor: int = field(default=5, metadata={"help": "Expansion factor for patch mixer."})
    dfactor: int = field(default=5, metadata={"help": "Expansion factor for embedding mixer."})
    wavelet_name: str = field(default="db2", metadata={"help": "Wavelet name (for full DWT port)."})
    level: int = field(default=1, metadata={"help": "Wavelet decomposition level (for full DWT port)."})

    # Patching
    patch_len: int = field(default=16, metadata={"help": "Patch length for token mixing."})
    patch_stride: int = field(default=8, metadata={"help": "Patch stride for token mixing."})

    # Integration flag
    no_decomposition: bool = field(
        default=True,
        metadata={"help": "If True, uses identity decomposition (no wavelet DWT port)."},
    )
    use_amp: bool = field(default=False, metadata={"help": "Reserved for full DWT port."})


__all__ = ["WPMixerConfig"]

