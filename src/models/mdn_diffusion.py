from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaLNResBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.cond = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, dim * 6))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.cond(cond).chunk(6, dim=-1)
        h = self.norm1(x)
        h = h * (1 + scale1) + shift1
        h = self.fc1(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        x = x + torch.tanh(gate1) * h

        h2 = self.norm2(x)
        h2 = h2 * (1 + scale2) + shift2
        h2 = F.silu(h2)
        x = x + torch.tanh(gate2) * h2
        return x


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.999).float()


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Sequence[int]) -> torch.Tensor:
    out = a.gather(-1, t)
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device) / max(1, half - 1))
    args = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class SCMDN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, n_blocks: int, dropout: float, class_dropout_prob: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.class_dropout_prob = class_dropout_prob
        self.null_class_idx = num_classes

        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.self_cond_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.time_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.SiLU(), nn.Linear(hidden_dim * 2, hidden_dim))
        self.class_embed = nn.Embedding(num_classes + 1, hidden_dim)
        self.blocks = nn.ModuleList([AdaLNResBlock(hidden_dim, hidden_dim, dropout=dropout) for _ in range(n_blocks)])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.final = nn.Linear(hidden_dim, input_dim)
        self.prototype_table = nn.Parameter(torch.randn(num_classes, input_dim) * 0.02)

    def maybe_drop_labels(self, labels: torch.Tensor, force_drop: bool = False) -> torch.Tensor:
        if force_drop:
            return torch.full_like(labels, self.null_class_idx)
        if (not self.training) or self.class_dropout_prob <= 0:
            return labels
        keep_mask = torch.rand(labels.shape, device=labels.device) > self.class_dropout_prob
        dropped = torch.full_like(labels, self.null_class_idx)
        return torch.where(keep_mask, labels, dropped)

    def forward(self, x_t: torch.Tensor, labels: torch.Tensor, timesteps: torch.Tensor, self_cond: Optional[torch.Tensor] = None, force_drop_label: bool = False) -> torch.Tensor:
        if self_cond is None:
            self_cond = torch.zeros_like(x_t)
        h = self.input_proj(x_t) + self.self_cond_proj(self_cond)
        t_emb = self.time_mlp(timestep_embedding(timesteps, self.hidden_dim))
        labels = self.maybe_drop_labels(labels, force_drop=force_drop_label)
        cond = t_emb + self.class_embed(labels)
        for block in self.blocks:
            h = block(h, cond)
        return self.final(self.final_norm(h))


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = {k: v.clone() for k, v in state_dict.items()}
