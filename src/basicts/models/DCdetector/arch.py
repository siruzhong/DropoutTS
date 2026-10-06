from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .config import DCdetectorConfig


class DCdetectorForReconstruction(nn.Module):
    """
    Dual-branch reconstruction baseline inspired by DCdetector.
    """

    def __init__(self, config: DCdetectorConfig):
        super().__init__()
        self.branch_agreement_weight = config.branch_agreement_weight
        self.temporal_position = nn.Parameter(torch.zeros(1, config.input_len, config.hidden_size))
        self.channel_position = nn.Parameter(torch.zeros(1, config.num_features, config.hidden_size))

        self.temporal_projection = nn.Linear(config.num_features, config.hidden_size)
        self.channel_projection = nn.Linear(config.input_len, config.hidden_size)

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        channel_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=config.num_layers)
        self.channel_encoder = nn.TransformerEncoder(channel_layer, num_layers=config.num_layers)
        self.temporal_head = nn.Linear(config.hidden_size, config.num_features)
        self.channel_head = nn.Linear(config.hidden_size, config.input_len)
        self.fusion = nn.Linear(config.num_features * 2, config.num_features)

    def forward(
            self,
            inputs: torch.Tensor,
            targets: Optional[torch.Tensor] = None,
            train: bool = False
            ):
        temporal_hidden = self.temporal_projection(inputs) + self.temporal_position[:, :inputs.shape[1], :]
        temporal_hidden = self.temporal_encoder(temporal_hidden)
        temporal_prediction = self.temporal_head(temporal_hidden)

        channel_inputs = inputs.transpose(1, 2)
        channel_hidden = self.channel_projection(channel_inputs) + self.channel_position[:, :inputs.shape[2], :]
        channel_hidden = self.channel_encoder(channel_hidden)
        channel_prediction = self.channel_head(channel_hidden).transpose(1, 2)

        prediction = self.fusion(torch.cat([temporal_prediction, channel_prediction], dim=-1))
        if train and targets is not None:
            recon_loss = F.mse_loss(prediction, targets)
            agreement_loss = F.mse_loss(temporal_prediction, channel_prediction.detach())
            return {
                "prediction": prediction,
                "loss": recon_loss + self.branch_agreement_weight * agreement_loss,
            }
        return prediction
