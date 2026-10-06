import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .config import AnomalyTransformerConfig


class AnomalyAttentionLayer(nn.Module):
    def __init__(self, config: AnomalyTransformerConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        if self.head_dim * config.num_heads != config.hidden_size:
            raise ValueError("hidden_size must be divisible by num_heads.")

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.log_sigma = nn.Parameter(torch.zeros(config.num_heads))
        self.dropout = nn.Dropout(config.dropout)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )

    def _prior(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        distance = (positions[None, :] - positions[:, None]).abs()
        sigma = F.softplus(self.log_sigma).to(device=device, dtype=dtype) + 1e-4
        prior = torch.exp(-0.5 * (distance[None, :, :] / sigma[:, None, None]) ** 2)
        prior = prior / prior.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return prior

    def _association_loss(self, series: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        prior = prior.unsqueeze(0).expand_as(series)
        series = series.clamp_min(1e-6)
        prior = prior.clamp_min(1e-6)
        return (
            F.kl_div(series.log(), prior, reduction="batchmean")
            + F.kl_div(prior.log(), series, reduction="batchmean")
        ) / series.shape[-2]

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        series = torch.softmax(scores, dim=-1)
        context = torch.matmul(self.dropout(series), value)
        context = context.transpose(1, 2).reshape(batch_size, seq_len, -1)
        hidden_states = self.norm1(hidden_states + self.o_proj(context))
        hidden_states = self.norm2(hidden_states + self.ffn(hidden_states))

        prior = self._prior(seq_len, hidden_states.device, hidden_states.dtype)
        assoc_loss = self._association_loss(series, prior)
        return hidden_states, assoc_loss


class AnomalyTransformerForReconstruction(nn.Module):
    """
    Reconstruction-compatible Anomaly Transformer baseline for BasicTS TSAD.
    """

    def __init__(self, config: AnomalyTransformerConfig):
        super().__init__()
        self.input_len = config.input_len
        self.association_loss_weight = config.association_loss_weight
        self.input_projection = nn.Linear(config.num_features, config.hidden_size)
        self.position_embedding = nn.Parameter(torch.zeros(1, config.input_len, config.hidden_size))
        self.layers = nn.ModuleList([AnomalyAttentionLayer(config) for _ in range(config.num_layers)])
        self.dropout = nn.Dropout(config.dropout)
        self.output_projection = nn.Linear(config.hidden_size, config.num_features)

    def forward(
            self,
            inputs: torch.Tensor,
            targets: Optional[torch.Tensor] = None,
            train: bool = False
            ):
        hidden_states = self.input_projection(inputs) + self.position_embedding[:, :inputs.shape[1], :]
        hidden_states = self.dropout(hidden_states)
        assoc_loss = inputs.new_tensor(0.0)
        for layer in self.layers:
            hidden_states, layer_loss = layer(hidden_states)
            assoc_loss = assoc_loss + layer_loss
        prediction = self.output_projection(hidden_states)

        if train and targets is not None:
            recon_loss = F.mse_loss(prediction, targets)
            loss = recon_loss + self.association_loss_weight * assoc_loss / max(1, len(self.layers))
            return {"prediction": prediction, "loss": loss}
        return prediction
