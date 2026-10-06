from __future__ import annotations

from typing import List

import torch
from torch import nn

from ..config import WPMixerConfig


class _IdentityDecomposition(nn.Module):
    """
    Minimal decomposition stub for integration.

    When `no_decomposition=True`, it behaves like an identity transform:
    - approximation coefficient series is the original input
    - detail coefficient list is empty
    """

    def __init__(self, input_length: int, pred_length: int, no_decomposition: bool = True):
        super().__init__()
        self.no_decomposition = no_decomposition
        if not self.no_decomposition:
            raise NotImplementedError(
                "Wavelet decomposition is not ported yet. "
                "Use `no_decomposition=True` in `WPMixerConfig` for a running baseline."
            )

        # (m+1) branches, i.e. approximation + m details
        self.input_w_dim = [input_length]
        self.pred_w_dim = [pred_length]

    def transform(self, x: torch.Tensor):
        # x: [B, C, L]
        return x, []

    def inv_transform(self, yl: torch.Tensor, yh: List[torch.Tensor]):
        # yl: [B, C, pred_L]
        return yl


class TokenMixer(nn.Module):
    def __init__(self, input_seq: int, pred_seq: int, dropout: float, factor: int, d_model: int):
        super().__init__()
        self.dropoutLayer = nn.Dropout(dropout)
        self.layers = nn.Sequential(
            nn.Linear(input_seq, pred_seq * factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pred_seq * factor, pred_seq),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, d_model, C, Patch_number]
        x = x.transpose(1, 2)  # -> [B, C, d_model, Patch_number]
        x = self.layers(x)  # linear acts on last dim (Patch_number)
        x = x.transpose(1, 2)  # -> [B, d_model, C, Patch_number]
        return x


class Mixer(nn.Module):
    def __init__(
        self,
        input_seq: int,
        out_seq: int,
        channel: int,
        d_model: int,
        dropout: float,
        tfactor: int,
        dfactor: int,
    ):
        super().__init__()
        self.input_seq = input_seq
        self.pred_seq = out_seq
        self.channel = channel
        self.d_model = d_model
        self.dropout = dropout
        self.tfactor = tfactor
        self.dfactor = dfactor

        self.tMixer = TokenMixer(
            input_seq=self.input_seq,
            pred_seq=self.pred_seq,
            dropout=self.dropout,
            factor=self.tfactor,
            d_model=self.d_model,
        )
        self.dropoutLayer = nn.Dropout(self.dropout)
        self.norm1 = nn.BatchNorm2d(self.channel)
        self.norm2 = nn.BatchNorm2d(self.channel)
        self.embeddingMixer = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * self.dfactor),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model * self.dfactor, self.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, Patch_number, d_model]
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)  # [B, d_model, C, Patch_number]
        x = self.dropoutLayer(self.tMixer(x))
        x = x.permute(0, 2, 3, 1)  # [B, C, Patch_number, d_model]
        x = self.norm2(x)
        x = x + self.dropoutLayer(self.embeddingMixer(x))
        return x


class ResolutionBranch(nn.Module):
    def __init__(
        self,
        input_seq: int,
        pred_seq: int,
        channel: int,
        d_model: int,
        dropout: float,
        embedding_dropout: float,
        tfactor: int,
        dfactor: int,
        patch_len: int,
        patch_stride: int,
    ):
        super().__init__()
        self.input_seq = input_seq
        self.pred_seq = pred_seq
        self.channel = channel
        self.d_model = d_model
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.tfactor = tfactor
        self.dfactor = dfactor
        self.patch_len = patch_len
        self.patch_stride = patch_stride

        self.patch_num = int((self.input_seq - self.patch_len) / self.patch_stride + 2)

        self.patch_norm = nn.BatchNorm2d(self.channel)
        self.patch_embedding_layer = nn.Linear(self.patch_len, self.d_model)
        self.mixer1 = Mixer(
            input_seq=self.patch_num,
            out_seq=self.patch_num,
            channel=self.channel,
            d_model=self.d_model,
            dropout=self.dropout,
            tfactor=self.tfactor,
            dfactor=self.dfactor,
        )
        self.mixer2 = Mixer(
            input_seq=self.patch_num,
            out_seq=self.patch_num,
            channel=self.channel,
            d_model=self.d_model,
            dropout=self.dropout,
            tfactor=self.tfactor,
            dfactor=self.dfactor,
        )
        self.norm = nn.BatchNorm2d(self.channel)
        self.dropoutLayer = nn.Dropout(self.embedding_dropout)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2, end_dim=-1),
            nn.Linear(self.patch_num * self.d_model, self.pred_seq),
        )

    def do_patching(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        x_end = x[:, :, -1:]
        x_padding = x_end.repeat(1, 1, self.patch_stride)
        x_new = torch.cat((x, x_padding), dim=-1)
        # -> [B, C, patch_num, patch_len]
        x_patch = x_new.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        return x_patch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, length_of_coefficient_series]
        x_patch = self.do_patching(x)
        x_patch = self.patch_norm(x_patch)
        x_emb = self.dropoutLayer(self.patch_embedding_layer(x_patch))
        out = self.mixer1(x_emb)
        res = out
        out = res + self.mixer2(out)
        out = self.norm(out)
        out = self.head(out)  # [B, C, pred_seq]
        return out


class WPMixerCore(nn.Module):
    def __init__(
        self,
        input_length: int,
        pred_length: int,
        level: int,
        channel: int,
        d_model: int,
        dropout: float,
        tfactor: int,
        dfactor: int,
        patch_len: int,
        patch_stride: int,
        no_decomposition: bool,
        wavelet_name: str = "db2",
        use_amp: bool = False,
    ):
        super().__init__()
        self.input_length = input_length
        self.pred_length = pred_length
        self.level = level
        self.channel = channel
        self.d_model = d_model
        self.dropout = dropout
        self.tfactor = tfactor
        self.dfactor = dfactor
        self.no_decomposition = no_decomposition
        self.wavelet_name = wavelet_name
        self.use_amp = use_amp

        self.Decomposition_model = _IdentityDecomposition(
            input_length=self.input_length,
            pred_length=self.pred_length,
            no_decomposition=no_decomposition,
        )
        self.input_w_dim = self.Decomposition_model.input_w_dim
        self.pred_w_dim = self.Decomposition_model.pred_w_dim

        self.patch_len = patch_len
        self.patch_stride = patch_stride

        self.resolutionBranch = nn.ModuleList(
            [
                ResolutionBranch(
                    input_seq=self.input_w_dim[i],
                    pred_seq=self.pred_w_dim[i],
                    channel=self.channel,
                    d_model=self.d_model,
                    dropout=self.dropout,
                    embedding_dropout=self.dropout,
                    tfactor=self.tfactor,
                    dfactor=self.dfactor,
                    patch_len=self.patch_len,
                    patch_stride=self.patch_stride,
                )
                for i in range(len(self.input_w_dim))
            ]
        )

    def forward(self, xL: torch.Tensor) -> torch.Tensor:
        # xL: [B, look_back_length, channel]
        x = xL.transpose(1, 2)  # [B, channel, look_back_length]

        xA, xD = self.Decomposition_model.transform(x)
        yA = self.resolutionBranch[0](xA)
        yD = []
        for i in range(len(xD)):
            yD_i = self.resolutionBranch[i + 1](xD[i])
            yD.append(yD_i)

        y = self.Decomposition_model.inv_transform(yA, yD)
        y = y.transpose(1, 2)  # [B, pred_length, channel]

        xT = y[:, -self.pred_length :, :]
        return xT


class WPMixerForForecasting(nn.Module):
    """
    WPMixer wrapper compatible with `basicts` runner.

    Expected input shape: [B, input_len, num_features]
    Returned prediction shape: [B, output_len, num_features]
    """

    def __init__(self, config: WPMixerConfig):
        super().__init__()
        self.output_len = config.output_len
        self.num_features = config.num_features
        self.wpmixerCore = WPMixerCore(
            input_length=config.input_len,
            pred_length=config.output_len,
            level=config.level,
            channel=config.num_features,
            d_model=config.d_model,
            dropout=config.dropout,
            tfactor=config.tfactor,
            dfactor=config.dfactor,
            patch_len=config.patch_len,
            patch_stride=config.patch_stride,
            no_decomposition=config.no_decomposition,
            wavelet_name=config.wavelet_name,
            use_amp=config.use_amp,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x_enc = inputs

        # Normalization (matches original WPMixer Model.forecast)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        pred = self.wpmixerCore(x_enc)  # [B, pred_len, C]
        pred = pred[:, :, -self.num_features :]

        # De-normalization
        dec_out = pred * (stdev[:, 0].unsqueeze(1).repeat(1, self.output_len, 1))
        dec_out = dec_out + (means[:, 0].unsqueeze(1).repeat(1, self.output_len, 1))
        return dec_out


__all__ = ["WPMixerForForecasting", "WPMixerCore"]

