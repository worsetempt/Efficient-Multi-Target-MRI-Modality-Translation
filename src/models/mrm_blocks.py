from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.GELU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class UpBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.GroupNorm(8, c_out),
            nn.GELU(),
            ResBlock(c_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


class SkipDecoder(nn.Module):
    def __init__(self, chs, out_ch: int = 1):
        super().__init__()
        c1, c2, c3, c4 = chs
        self.p4 = nn.Sequential(nn.Conv2d(c4, 512, 1), nn.GroupNorm(8, 512), nn.GELU())
        self.up4 = UpBlock(512, 256)
        self.fuse3 = nn.Sequential(
            nn.Conv2d(256 + c3, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            ResBlock(256),
        )
        self.up3 = UpBlock(256, 128)
        self.fuse2 = nn.Sequential(
            nn.Conv2d(128 + c2, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            ResBlock(128),
        )
        self.up2 = UpBlock(128, 64)
        self.fuse1 = nn.Sequential(
            nn.Conv2d(64 + c1, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            ResBlock(64),
        )
        self.up1 = UpBlock(64, 32)
        self.up0 = UpBlock(32, 16)
        self.out = nn.Conv2d(16, out_ch, 3, padding=1)

    def forward(self, f1, f2, f3, f4):
        x = self.p4(f4)
        x = self.up4(x)
        x = torch.cat([x, f3], dim=1)
        x = self.fuse3(x)
        x = self.up3(x)
        x = torch.cat([x, f2], dim=1)
        x = self.fuse2(x)
        x = self.up2(x)
        x = torch.cat([x, f1], dim=1)
        x = self.fuse1(x)
        x = self.up1(x)
        x = self.up0(x)
        return self.out(x)
