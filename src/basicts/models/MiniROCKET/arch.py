import math

import torch
from torch import nn

from .config import MiniROCKETConfig


class MiniROCKETForClassification(nn.Module):
    """
    MiniROCKET-style fixed convolutional feature extractor with a trainable linear head.
    """

    def __init__(self, config: MiniROCKETConfig):
        super().__init__()
        self.config = config
        specs = [
            (kernel, dilation)
            for kernel in config.kernel_sizes
            for dilation in config.dilations
            if (kernel - 1) * dilation < config.input_len
        ]
        if not specs:
            specs = [(min(config.input_len, 7), 1)]

        kernels_per_spec = max(1, math.ceil(config.num_kernels / len(specs)))
        generator = torch.Generator()
        generator.manual_seed(config.seed)
        convs = []
        total_kernels = 0
        for kernel, dilation in specs:
            out_channels = min(kernels_per_spec, config.num_kernels - total_kernels)
            if out_channels <= 0:
                break
            conv = nn.Conv1d(
                config.num_features,
                out_channels,
                kernel_size=kernel,
                dilation=dilation,
                padding=((kernel - 1) * dilation) // 2,
                bias=True,
            )
            with torch.no_grad():
                weights = torch.randint(
                    low=0,
                    high=2,
                    size=conv.weight.shape,
                    generator=generator,
                    dtype=torch.float32,
                )
                weights = weights.mul_(3.0).sub_(1.0)
                weights = weights - weights.mean(dim=-1, keepdim=True)
                conv.weight.copy_(weights)
                conv.bias.uniform_(-1.0, 1.0, generator=generator)
            for param in conv.parameters():
                param.requires_grad = False
            convs.append(conv)
            total_kernels += out_channels

        self.convs = nn.ModuleList(convs)
        self.dropout = nn.Dropout(config.dropout)
        self.classification_head = nn.Linear(total_kernels * 2, config.num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(-1)
        inputs = inputs.transpose(1, 2)
        features = []
        for conv in self.convs:
            outputs = conv(inputs)
            features.append((outputs > 0).float().mean(dim=-1))
            features.append(outputs.amax(dim=-1))
        features = self.dropout(torch.cat(features, dim=-1))
        return self.classification_head(features)
