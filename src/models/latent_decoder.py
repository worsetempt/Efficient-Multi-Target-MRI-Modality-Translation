from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


def make_gn_groups(c: int, max_groups: int = 8) -> int:
    g = min(max_groups, c)
    while g > 1 and c % g != 0:
        g -= 1
    return g


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        g = make_gn_groups(c)
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(g, c),
            nn.GELU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(g, c),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class UpBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, num_res: int = 1):
        super().__init__()
        g = make_gn_groups(c_out)
        layers: List[nn.Module] = [
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.GroupNorm(g, c_out),
            nn.GELU(),
        ]
        for _ in range(num_res):
            layers.append(ResBlock(c_out))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RefinementHead(nn.Module):
    def __init__(self, c: int, out_ch: int = 1):
        super().__init__()
        g = make_gn_groups(c)
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(g, c),
            nn.GELU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(g, c),
            nn.GELU(),
            nn.Conv2d(c, out_ch, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LatentDecoderV2(nn.Module):
    def __init__(self, latent_dim: int = 768, base_ch: int = 384, out_ch: int = 1):
        super().__init__()
        self.latent_dim = latent_dim
        self.base_ch = base_ch

        c0 = base_ch
        c1 = max(base_ch // 2, 64)
        c2 = max(base_ch // 4, 64)
        c3 = max(base_ch // 8, 48)
        c4 = max(base_ch // 16, 32)
        c5 = max(base_ch // 24, 24)

        g0 = make_gn_groups(c0)
        self.fc = nn.Linear(latent_dim, c0 * 7 * 7)
        self.block0 = nn.Sequential(
            nn.Conv2d(c0, c0, 3, padding=1),
            nn.GroupNorm(g0, c0),
            nn.GELU(),
            ResBlock(c0),
            ResBlock(c0),
        )
        self.up4 = UpBlock(c0, c1, num_res=2)
        self.up3 = UpBlock(c1, c2, num_res=2)
        self.up2 = UpBlock(c2, c3, num_res=2)
        self.up1 = UpBlock(c3, c4, num_res=1)
        self.up0 = UpBlock(c4, c5, num_res=1)
        self.refine = RefinementHead(c5, out_ch=out_ch)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2:
            z = z.view(z.size(0), -1)
        x = self.fc(z)
        x = x.view(z.size(0), self.base_ch, 7, 7)
        x = self.block0(x)
        x = self.up4(x)
        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)
        x = self.up0(x)
        return self.refine(x)
