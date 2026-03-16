from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float, use_layernorm: bool):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim) if use_layernorm else nn.Identity()
        self.norm2 = nn.LayerNorm(dim) if use_layernorm else nn.Identity()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.fc2(self.dropout(F.gelu(self.fc1(h))))
        x = x + self.dropout(h)
        h = self.norm2(x)
        h = self.fc2(self.dropout(F.gelu(self.fc1(h))))
        return x + self.dropout(h)


class LatentTranslator(nn.Module):
    def __init__(self, latent_dim: int, num_modalities: int, hidden_dim: int, n_blocks: int, dropout: float, use_layernorm: bool = True):
        super().__init__()
        embed_dim = hidden_dim // 4
        self.src_embed = nn.Embedding(num_modalities, embed_dim)
        self.tgt_embed = nn.Embedding(num_modalities, embed_dim)
        self.input_proj = nn.Linear(latent_dim + 2 * embed_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout, use_layernorm) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
        self.delta_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_src: torch.Tensor, src_id: torch.Tensor, tgt_id: torch.Tensor):
        x = torch.cat([z_src, self.src_embed(src_id), self.tgt_embed(tgt_id)], dim=-1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        delta = self.delta_head(x)
        return z_src + delta, delta
