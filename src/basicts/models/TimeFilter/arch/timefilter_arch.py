from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.timefilter_config import TimeFilterConfig


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class Normalize(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False, subtract_last: bool = False, non_norm: bool = False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def _init_params(self) -> None:
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        raise NotImplementedError

    def _get_statistics(self, x: torch.Tensor) -> None:
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.non_norm:
            return x
        # Proper inverse of `_normalize`:
        # - `_normalize` always divides by `self.stdev`
        # - if `affine=True`, `_normalize` also applies affine transform
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            return x + self.last
        return x + self.mean


class PatchEmbed(nn.Module):
    def __init__(self, dim: int, patch_len: int, stride: Optional[int] = None, pos: bool = True) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = patch_len if stride is None else stride
        self.patch_proj = nn.Linear(self.patch_len, dim)
        self.pos = pos
        if self.pos:
            pos_emb_theta = 10000
            self.pe = PositionalEmbedding(dim, pos_emb_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, T] in the reference repo, but we also support x: [B, T] (N=1)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x: [B, N*L, P] (or [B, L, P] for 2D input)
        x = self.patch_proj(x)
        if self.pos:
            x = x + self.pe(x)
        return x


def mask_topk(x: torch.Tensor, alpha: float = 0.5, largest: bool = False) -> torch.Tensor:
    # x: [B, H, L, L]
    k = int(alpha * x.shape[-1])
    _, topk_indices = torch.topk(x, k, dim=-1, largest=largest)
    mask = torch.ones_like(x, dtype=torch.float32)
    # 1 is topk -> set to 0 (mask it out)
    mask.scatter_(-1, topk_indices, 0)
    return mask


class GCN(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.n_heads = n_heads

    def forward(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # adj [B, H, L, L], x [B, L, D]
        B, L, D = x.shape
        x = self.proj(x).view(B, L, self.n_heads, -1)  # [B, L, H, D_]
        adj = F.normalize(adj, p=1, dim=-1)
        x = torch.einsum("bhij,bjhd->bihd", adj, x).contiguous()  # [B, H, L, D_] (layout after einsum)
        x = x.view(B, L, -1)
        return x


class mask_moe(nn.Module):
    def __init__(self, n_vars: int, top_p: float = 0.5, num_experts: int = 3, in_dim: int = 96):
        super().__init__()
        self.num_experts = num_experts
        self.n_vars = n_vars
        self.in_dim = in_dim

        self.gate = nn.Linear(self.in_dim, num_experts, bias=False)
        self.noise = nn.Linear(self.in_dim, num_experts, bias=False)
        self.noisy_gating = 1
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(2)
        self.top_p = top_p

    def cv_squared(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def cross_entropy(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return -torch.mul(x, torch.log(x + eps)).sum(dim=1).mean()

    def noisy_top_k_gating(self, x: torch.Tensor, is_training: bool, noise_epsilon: float = 1e-2):
        clean_logits = self.gate(x)
        if self.noisy_gating and is_training:
            raw_noise = self.noise(x)
            noise_stddev = self.softplus(raw_noise) + noise_epsilon
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            logits = noisy_logits
        else:
            logits = clean_logits

        logits = self.softmax(logits)
        loss_dynamic = self.cross_entropy(logits)

        sorted_probs, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs > self.top_p

        threshold_indices = mask.long().argmax(dim=-1)
        threshold_mask = torch.nn.functional.one_hot(threshold_indices, num_classes=sorted_indices.size(-1)).bool()
        mask = mask & ~threshold_mask

        top_p_mask = torch.zeros_like(mask)
        zero_indices = (mask == 0).nonzero(as_tuple=True)
        top_p_mask[
            zero_indices[0], zero_indices[1], sorted_indices[zero_indices[0], zero_indices[1], zero_indices[2]]
        ] = 1

        sorted_probs = torch.where(mask, 0.0, sorted_probs)
        loss_importance = self.cv_squared(sorted_probs.sum(0))
        lambda_2 = 0.1
        loss = loss_importance + lambda_2 * loss_dynamic

        return top_p_mask, loss

    def forward(self, x: torch.Tensor, masks=None):
        # x: [B, H, L, L]
        B, H, L, _ = x.shape
        device = x.device
        dtype = torch.float32

        mask_base = torch.eye(L, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)  # [1,1,L,L]
        if self.top_p == 0.0:
            return mask_base, 0.0

        # reshape to [B*H, L, L]
        x = x.reshape(B * H, L, L)
        gates, loss = self.noisy_top_k_gating(x, self.training)
        gates = gates.reshape(B, H, L, -1).float()  # [B,H,L,3]

        if masks is None:
            masks = []
            N = L // self.n_vars
            for k in range(L):
                S = ((torch.arange(L, device=device) % N == k % N) & (torch.arange(L, device=device) != k)).to(dtype)
                T = (
                    (torch.arange(L, device=device) >= k // N * N)
                    & (torch.arange(L, device=device) < k // N * N + N)
                    & (torch.arange(L, device=device) != k)
                ).to(dtype)
                ST = torch.ones(L, device=device, dtype=dtype) - S - T
                masks.append(torch.stack([S, T, ST], dim=0))
            masks = torch.stack(masks, dim=0)  # [L,3,L]

        # masks: [L,3,L]
        mask = torch.einsum("bhli,lid->bhld", gates, masks) + mask_base
        return mask, loss


class GraphLearner(nn.Module):
    def __init__(self, dim: int, n_vars: int, top_p: float = 0.5, in_dim: int = 96):
        super().__init__()
        self.dim = dim
        self.proj_1 = nn.Linear(dim, dim)
        self.proj_2 = nn.Linear(dim, dim)
        self.n_vars = n_vars
        self.mask_moe = mask_moe(n_vars, top_p=top_p, in_dim=in_dim)

    def forward(self, x: torch.Tensor, masks=None, alpha: float = 0.5):
        # x: [B, H, L, D]
        adj = F.gelu(torch.einsum("bhid,bhjd->bhij", self.proj_1(x), self.proj_2(x)))
        adj = adj * mask_topk(adj, alpha)  # KNN-like sparsification (reference behaviour)
        mask, loss = self.mask_moe(adj, masks)
        adj = adj * mask
        return adj, loss  # [B, H, L, L], loss


class GraphFilter(nn.Module):
    def __init__(self, dim: int, n_vars: int, n_heads: int = 4, scale=None, top_p: float = 0.5, dropout: float = 0.0, in_dim: int = 96):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.scale = dim ** (-0.5) if scale is None else scale
        self.dropout = nn.Dropout(dropout)
        self.graph_learner = GraphLearner(self.dim // self.n_heads, n_vars, top_p=top_p, in_dim=in_dim)
        self.graph_conv = GCN(self.dim, self.n_heads)

    def forward(self, x: torch.Tensor, masks=None, alpha: float = 0.5):
        # x: [B, L, D]
        B, L, D = x.shape
        adj, loss = self.graph_learner(x.reshape(B, L, self.n_heads, -1).permute(0, 2, 1, 3), masks, alpha)  # [B,H,L,L]
        adj = torch.softmax(adj, dim=-1)
        adj = self.dropout(adj)
        out = self.graph_conv(adj, x)  # [B,L,D]
        return out, loss


class GraphBlock(nn.Module):
    def __init__(self, dim: int, n_vars: int, d_ff: Optional[int] = None, n_heads: int = 4, top_p: float = 0.5, dropout: float = 0.0, in_dim: int = 96):
        super().__init__()
        self.dim = dim
        self.d_ff = dim * 4 if d_ff is None else d_ff
        self.gnn = GraphFilter(self.dim, n_vars, n_heads, top_p=top_p, dropout=dropout, in_dim=in_dim)
        self.norm1 = nn.LayerNorm(self.dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, self.d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_ff, self.dim),
        )
        self.norm2 = nn.LayerNorm(self.dim)

    def forward(self, x: torch.Tensor, masks=None, alpha: float = 0.5):
        out, loss = self.gnn(self.norm1(x), masks, alpha)
        x = x + out
        x = x + self.ffn(self.norm2(x))
        return x, loss


class TimeFilter_Backbone(nn.Module):
    def __init__(self, hidden_dim: int, n_vars: int, d_ff: Optional[int] = None, n_heads: int = 4, n_blocks: int = 3, top_p: float = 0.5, dropout: float = 0.0, in_dim: int = 96):
        super().__init__()
        self.dim = hidden_dim
        self.d_ff = self.dim * 2 if d_ff is None else d_ff
        self.blocks = nn.ModuleList(
            [GraphBlock(self.dim, n_vars, self.d_ff, n_heads, top_p, dropout, in_dim) for _ in range(n_blocks)]
        )
        self.n_blocks = n_blocks

    def forward(self, x: torch.Tensor, masks=None, alpha: float = 0.5):
        # x: [B, L, D]
        moe_loss = 0.0
        for block in self.blocks:
            x, loss = block(x, masks, alpha)
            moe_loss += loss
        moe_loss = moe_loss / self.n_blocks
        return x, moe_loss


class TimeFilterModel(nn.Module):
    def __init__(self, configs: SimpleNamespace):
        super().__init__()
        self.args = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_vars = configs.c_out
        self.dim = configs.d_model
        self.d_ff = configs.d_ff
        self.patch_len = configs.patch_len

        self.stride = self.patch_len
        self.num_patches = int((self.seq_len - self.patch_len) / self.stride + 1)  # per-variable patches

        # Filter
        self.alpha = 0.1 if configs.alpha is None else configs.alpha
        self.top_p = 0.5 if configs.top_p is None else configs.top_p

        # embed
        self.patch_embed = PatchEmbed(self.dim, self.patch_len, self.stride, configs.pos)

        # TimeFilter.sh Backbone
        self.backbone = TimeFilter_Backbone(
            self.dim,
            self.n_vars,
            self.d_ff,
            configs.n_heads,
            configs.e_layers,
            self.top_p,
            configs.dropout,
            self.seq_len * self.n_vars // self.patch_len,
        )

        if self.task_name in {"long_term_forecast", "short_term_forecast"}:
            self.head = nn.Linear(self.dim * self.num_patches, self.pred_len)
        elif self.task_name in {"imputation", "anomaly_detection"}:
            self.head = nn.Linear(self.dim * self.num_patches, self.seq_len)
        elif self.task_name == "classification":
            self.num_patches = int((self.seq_len * configs.enc_in - self.patch_len) / self.stride + 1)  # L
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(self.dim * self.num_patches, configs.num_class)

        # Without RevIN (reference flag)
        self.use_RevIN = False
        self.norm = Normalize(configs.enc_in, affine=self.use_RevIN)

        self._mask_cache: dict[str, torch.Tensor] = {}

    def _get_mask(self, device: torch.device) -> torch.Tensor:
        # Cache masks per device to avoid rebuilding in every forward.
        cache_key = str(device)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        dtype = torch.float32
        L = self.args.seq_len * self.args.c_out // self.args.patch_len
        N = self.args.seq_len // self.args.patch_len

        masks = []
        for k in range(L):
            ar = torch.arange(L, device=device)
            S = ((ar % N == k % N) & (ar != k)).to(dtype)
            T = ((ar >= k // N * N) & (ar < k // N * N + N) & (ar != k)).to(dtype)
            ST = torch.ones(L, device=device, dtype=dtype) - S - T
            ST[k] = 0.0
            masks.append(torch.stack([S, T, ST], dim=0))

        masks_tensor = torch.stack(masks, dim=0)  # [L,3,L]
        self._mask_cache[cache_key] = masks_tensor
        return masks_tensor

    def forecast(self, x: torch.Tensor, x_dec=None, x_mark_dec=None):
        # x: [B, T, C]
        x = self.norm(x, "norm")

        # x -> [B, C*T] then patchify along the last dim
        B, T, C = x.shape
        x = x.permute(0, 2, 1).reshape(-1, C * T)  # [B, C*T]
        x = self.patch_embed(x)  # [B, L, D]

        masks = self._get_mask(x.device)
        x, _moe_loss = self.backbone(x, masks, self.alpha)

        x = self.head(x.reshape(-1, self.n_vars, self.num_patches, self.dim).flatten(start_dim=-2))  # [B,C,pred_len]
        x = x.permute(0, 2, 1)  # [B,pred_len,C]
        x = self.norm(x, "denorm")
        return x

    def forward(self, x_enc: torch.Tensor, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in {"long_term_forecast", "short_term_forecast"}:
            dec_out = self.forecast(x_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]

        if self.task_name == "imputation":
            raise NotImplementedError("TimeFilter imputation task is not wrapped for basicts yet.")
        if self.task_name == "anomaly_detection":
            raise NotImplementedError("TimeFilter anomaly_detection task is not wrapped for basicts yet.")
        if self.task_name == "classification":
            raise NotImplementedError("TimeFilter classification task is not wrapped for basicts yet.")
        return None


class TimeFilterForForecasting(nn.Module):
    """
    basicts-style wrapper around the upstream TimeFilter implementation.

    Expected input:
      - `inputs`: [B, input_len, num_features]
    Output:
      - [B, output_len, num_features]
    """

    def __init__(self, config: TimeFilterConfig):
        super().__init__()
        self.output_len = config.output_len
        self.num_features = config.num_features

        up_cfg = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=config.input_len,
            pred_len=config.output_len,
            c_out=config.num_features,
            enc_in=config.num_features,
            d_model=config.d_model,
            d_ff=config.d_ff,
            patch_len=config.patch_len,
            alpha=config.alpha,
            top_p=config.top_p,
            pos=config.pos,
            n_heads=config.n_heads,
            e_layers=config.e_layers,
            dropout=config.dropout,
            # classification-only placeholders
            enc_in_ignored=config.num_features,
            num_class=1,
        )
        self.model = TimeFilterModel(up_cfg)

    def forward(self, inputs: torch.Tensor, inputs_timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        # TimeFilter does not use timestamps; keep signature for runner compatibility.
        _ = inputs_timestamps
        return self.model(inputs, None, None, None)


__all__ = ["TimeFilterForForecasting"]

