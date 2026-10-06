from __future__ import annotations

from math import sqrt
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..config import MultiPatchFormerConfig


def _activation(name: str):
    return F.relu if name == "relu" else F.gelu


class FullAttention(nn.Module):
    def __init__(
        self,
        mask_flag: bool = True,
        scale: Optional[float] = None,
        attention_dropout: float = 0.1,
        output_attention: bool = False,
    ) -> None:
        super().__init__()
        self.mask_flag = mask_flag
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, q_len, num_heads, head_dim = queries.shape
        _, k_len, _, _ = keys.shape
        scale = self.scale or 1.0 / sqrt(head_dim)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = torch.triu(
                    torch.ones(q_len, k_len, dtype=torch.bool, device=queries.device),
                    diagonal=1,
                )
            scores = scores.masked_fill(attn_mask, -torch.inf)

        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        output = torch.einsum("bhls,bshd->blhd", attn, values)
        return output.contiguous(), attn if self.output_attention else None


class AttentionLayer(nn.Module):
    def __init__(self, attention: FullAttention, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.inner_attention = attention
        self.n_heads = n_heads
        self.d_keys = d_model // n_heads
        self.d_values = d_model // n_heads
        self.query_projection = nn.Linear(d_model, self.d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, self.d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, self.d_values * n_heads)
        self.out_projection = nn.Linear(self.d_values * n_heads, d_model)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, q_len, _ = queries.shape
        _, k_len, _ = keys.shape

        queries = self.query_projection(queries).view(batch_size, q_len, self.n_heads, self.d_keys)
        keys = self.key_projection(keys).view(batch_size, k_len, self.n_heads, self.d_keys)
        values = self.value_projection(values).view(batch_size, k_len, self.n_heads, self.d_values)

        output, attn = self.inner_attention(queries, keys, values, attn_mask)
        output = output.view(batch_size, q_len, -1)
        return self.out_projection(output), attn


class Transpose(nn.Module):
    def __init__(self, *dims: int, contiguous: bool = False) -> None:
        super().__init__()
        self.dims = dims
        self.contiguous = contiguous

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(*self.dims)
        return x.contiguous() if self.contiguous else x


class EncoderLayer(nn.Module):
    def __init__(self, attention: AttentionLayer, d_model: int, d_ff: int, dropout: float, activation: str) -> None:
        super().__init__()
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _activation(activation)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, norm_layer: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.layers = layers
        self.norm = norm_layer

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, list[Optional[torch.Tensor]]]:
        attns = []
        for layer in self.layers:
            x, attn = layer(x, attn_mask=attn_mask)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float, activation: str) -> None:
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Dropout(dropout),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MultiPatchFormerForForecasting(nn.Module):
    """
    MultiPatchFormer adapted from THUML Time-Series-Library.

    Source: https://github.com/thuml/Time-Series-Library/blob/main/models/MultiPatchFormer.py
    """

    patch_len1 = 8
    patch_len2 = 16
    patch_len3 = 24
    patch_len4 = 32
    stride1 = 8
    stride2 = 8
    stride3 = 7
    stride4 = 6

    def __init__(self, config: MultiPatchFormerConfig):
        super().__init__()
        self.seq_len = config.input_len
        self.pred_len = config.output_len
        self.enc_in = config.num_features
        self.d_model = config.d_model
        self.output_segment_len = self.pred_len // 8
        self.output_last_len = self.pred_len - self.output_segment_len * 7
        self.output_base_len = self.output_segment_len * 3 + self.output_last_len

        self.patch_num1 = int((self.seq_len + self.stride1 - self.patch_len1) / self.stride1) + 1
        self.patch_num2 = int((self.seq_len - self.patch_len2) / self.stride2) + 2
        self.patch_num3 = int((self.seq_len - self.patch_len3) / self.stride3) + 2
        self.patch_num4 = int((self.seq_len - self.patch_len4) / self.stride4) + 2
        self.padding_patch_layer1 = nn.ReplicationPad1d((0, self.stride1))
        self.padding_patch_layer2 = nn.ReplicationPad1d((0, self.stride2))
        self.padding_patch_layer3 = nn.ReplicationPad1d((0, self.stride3))
        self.padding_patch_layer4 = nn.ReplicationPad1d((0, self.stride4))

        self.in_layer1 = nn.Linear(self.patch_len1, self.d_model)
        self.in_layer2 = nn.Linear(self.patch_len2, self.d_model)
        self.in_layer3 = nn.Linear(self.patch_len3, self.d_model)
        self.in_layer4 = nn.Linear(self.patch_len4, self.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.encoder1 = Encoder(
            nn.ModuleList([
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            attention_dropout=config.dropout,
                            output_attention=config.output_attention,
                        ),
                        config.d_model,
                        config.n_heads,
                    ),
                    config.d_model,
                    config.d_ff,
                    config.dropout,
                    config.activation,
                )
                for _ in range(config.e_layers)
            ]),
            norm_layer=torch.nn.LayerNorm(config.d_model),
        )
        self.encoder2 = Encoder(
            nn.ModuleList([
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            attention_dropout=config.dropout,
                            output_attention=config.output_attention,
                        ),
                        config.d_model,
                        config.n_heads,
                    ),
                    config.d_model,
                    config.d_ff,
                    config.dropout,
                    config.activation,
                )
                for _ in range(config.e_layers)
            ]),
            norm_layer=torch.nn.LayerNorm(config.d_model),
        )
        self.encoder3 = Encoder(
            nn.ModuleList([
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            attention_dropout=config.dropout,
                            output_attention=config.output_attention,
                        ),
                        config.d_model,
                        config.n_heads,
                    ),
                    config.d_model,
                    config.d_ff,
                    config.dropout,
                    config.activation,
                )
                for _ in range(config.e_layers)
            ]),
            norm_layer=torch.nn.LayerNorm(config.d_model),
        )

        self.out_layer1 = nn.Linear(self.d_model * self.patch_num1, self.output_segment_len)
        self.out_layer2 = nn.Linear(self.d_model * self.patch_num2, self.output_segment_len)
        self.out_layer3 = nn.Linear(self.d_model * self.patch_num3, self.output_segment_len)
        self.out_layer4 = nn.Linear(self.d_model * self.patch_num4, self.output_last_len)
        self.out_layer5 = nn.Linear(self.output_base_len * self.enc_in, self.output_segment_len * self.enc_in)
        self.out_layer6 = nn.Linear(self.output_base_len * self.enc_in, self.output_segment_len * self.enc_in)
        self.out_layer7 = nn.Linear(self.output_base_len * self.enc_in, self.output_segment_len * self.enc_in)
        self.out_layer8 = nn.Linear(self.output_base_len * self.enc_in, self.output_last_len * self.enc_in)
        self.ff = FeedForward(self.output_base_len, config.d_ff, config.dropout, config.activation)

        self.dropout1 = nn.Dropout(config.dropout)
        self.norm1 = nn.Sequential(
            Transpose(1, 2),
            nn.BatchNorm1d(self.output_base_len),
            Transpose(1, 2),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        inputs_timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        means = inputs.mean(1, keepdim=True).detach()
        inputs = inputs - means
        stdev = torch.sqrt(torch.var(inputs, dim=1, keepdim=True, unbiased=False) + 1e-5)
        inputs = inputs / stdev

        inputs = inputs.permute(0, 2, 1)
        batch_size = inputs.shape[0]

        x1 = self.padding_patch_layer1(inputs).unfold(dimension=-1, size=self.patch_len1, step=self.stride1)
        x1 = self.dropout(self.in_layer1(x1))
        x2 = self.padding_patch_layer2(inputs).unfold(dimension=-1, size=self.patch_len2, step=self.stride2)
        x2 = self.dropout(self.in_layer2(x2))
        x3 = self.padding_patch_layer3(inputs).unfold(dimension=-1, size=self.patch_len3, step=self.stride3)
        x3 = self.dropout(self.in_layer3(x3))
        x4 = self.padding_patch_layer4(inputs).unfold(dimension=-1, size=self.patch_len4, step=self.stride4)
        x4 = self.dropout(self.in_layer4(x4))
        p1, p2, p3, p4 = x1.shape[2], x2.shape[2], x3.shape[2], x4.shape[2]

        x = torch.cat([x1, x2, x3, x4], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x, _ = self.encoder1(x)
        x = torch.reshape(x, (-1, self.enc_in, x.shape[-2], x.shape[-1]))
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x, _ = self.encoder2(x)
        x = torch.reshape(x, (-1, p1 + p2 + p3 + p4, x.shape[-2], x.shape[-1]))
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x, _ = self.encoder3(x)
        x = torch.reshape(x, (-1, self.enc_in, x.shape[-2], x.shape[-1]))

        x1 = x[:, :, 0:p1, :].reshape(batch_size, self.enc_in, -1)
        x2 = x[:, :, p1 : p1 + p2, :].reshape(batch_size, self.enc_in, -1)
        x3 = x[:, :, p1 + p2 : p1 + p2 + p3, :].reshape(
            batch_size,
            self.enc_in,
            -1,
        )
        x4 = x[:, :, p1 + p2 + p3 : p1 + p2 + p3 + p4, :].reshape(batch_size, self.enc_in, -1)

        y1 = self.out_layer1(x1)
        y2 = self.out_layer2(x2)
        y3 = self.out_layer3(x3)
        y4 = self.out_layer4(x4)

        y5 = torch.cat([y1, y2, y3, y4], dim=2)
        y5 = self.norm1(y5)
        y5 = self.dropout1(self.ff(y5)) + y5

        y5 = y5.reshape(batch_size, -1)
        y5 = self.out_layer5(y5).reshape(batch_size, self.enc_in, -1)
        y5 = y5 + y1

        y6 = torch.cat([y1, y2, y3, y4], dim=2)
        y6 = self.norm1(y6)
        y6 = self.dropout1(self.ff(y6)) + y6
        y6 = y6.reshape(batch_size, -1)
        y6 = self.out_layer6(y6).reshape(batch_size, self.enc_in, -1)
        y6 = y6 + y2

        y7 = torch.cat([y1, y2, y3, y4], dim=2)
        y7 = self.norm1(y7)
        y7 = self.dropout1(self.ff(y7)) + y7
        y7 = y7.reshape(batch_size, -1)
        y7 = self.out_layer7(y7).reshape(batch_size, self.enc_in, -1)
        y7 = y7 + y3

        y8 = torch.cat([y1, y2, y3, y4], dim=2)
        y8 = self.norm1(y8)
        y8 = self.dropout1(self.ff(y8)) + y8
        y8 = y8.reshape(batch_size, -1)
        y8 = self.out_layer8(y8).reshape(batch_size, self.enc_in, -1)
        y8 = y8 + y4

        prediction = torch.cat([y1, y5, y2, y6, y3, y7, y4, y8], dim=2)
        prediction = prediction[:, :, : self.pred_len].permute(0, 2, 1)
        prediction = prediction * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        prediction = prediction + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return prediction


__all__ = ["MultiPatchFormerForForecasting"]
