from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .config import TranADConfig


class TranADForReconstruction(nn.Module):
    """
    Two-stage Transformer reconstruction baseline inspired by TranAD.
    """

    def __init__(self, config: TranADConfig):
        super().__init__()
        self.first_stage_loss_weight = config.first_stage_loss_weight
        self.position_embedding = nn.Parameter(torch.zeros(1, config.input_len, config.hidden_size))
        self.input_projection = nn.Linear(config.num_features, config.hidden_size)
        self.feedback_projection = nn.Linear(config.num_features * 2, config.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.output_projection = nn.Linear(config.hidden_size, config.num_features)

    def _encode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.encoder(hidden_states + self.position_embedding[:, :hidden_states.shape[1], :])

    def forward(
            self,
            inputs: torch.Tensor,
            targets: Optional[torch.Tensor] = None,
            train: bool = False
            ):
        first_hidden = self._encode(self.input_projection(inputs))
        first_prediction = self.output_projection(first_hidden)
        focus = (first_prediction.detach() - inputs).pow(2)
        second_inputs = torch.cat([inputs, focus], dim=-1)
        second_hidden = self._encode(self.feedback_projection(second_inputs))
        prediction = self.output_projection(second_hidden)

        if train and targets is not None:
            loss = (
                F.mse_loss(prediction, targets)
                + self.first_stage_loss_weight * F.mse_loss(first_prediction, targets)
            )
            return {"prediction": prediction, "loss": loss}
        return prediction
