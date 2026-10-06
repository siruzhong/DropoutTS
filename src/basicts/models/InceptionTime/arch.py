import torch
from torch import nn

from .config import InceptionTimeConfig


class InceptionModule(nn.Module):
    def __init__(self, in_channels: int, config: InceptionTimeConfig):
        super().__init__()
        bottleneck_channels = min(config.bottleneck_size, in_channels)
        self.bottleneck = (
            nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            if in_channels > 1
            else nn.Identity()
        )
        branch_in = bottleneck_channels if in_channels > 1 else in_channels
        self.branches = nn.ModuleList([
            nn.Conv1d(
                branch_in,
                config.hidden_size,
                kernel_size=k,
                padding=k // 2,
                bias=False,
            )
            for k in config.kernel_sizes
        ])
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, config.hidden_size, kernel_size=1, bias=False),
        )
        out_channels = config.hidden_size * (len(config.kernel_sizes) + 1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        bottleneck = self.bottleneck(inputs)
        outputs = [branch(bottleneck) for branch in self.branches]
        outputs.append(self.pool_branch(inputs))
        outputs = torch.cat(outputs, dim=1)
        return self.dropout(self.activation(self.norm(outputs)))


class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, config: InceptionTimeConfig, use_residual: bool):
        super().__init__()
        self.module = InceptionModule(in_channels, config)
        out_channels = config.hidden_size * (len(config.kernel_sizes) + 1)
        self.use_residual = use_residual
        if use_residual:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.module(inputs)
        if self.use_residual:
            outputs = self.activation(outputs + self.shortcut(inputs))
        return outputs


class InceptionTimeForClassification(nn.Module):
    """
    InceptionTime classifier for time series classification benchmarks.
    """

    def __init__(self, config: InceptionTimeConfig):
        super().__init__()
        channels = config.num_features
        blocks = []
        for idx in range(config.num_blocks):
            block = InceptionBlock(
                channels,
                config,
                use_residual=(idx % 3 == 2),
            )
            blocks.append(block)
            channels = config.hidden_size * (len(config.kernel_sizes) + 1)
        self.backbone = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classification_head = nn.Linear(channels, config.num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(-1)
        hidden_states = self.backbone(inputs.transpose(1, 2))
        pooled = self.pool(hidden_states).squeeze(-1)
        return self.classification_head(pooled)
