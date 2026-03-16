from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_c, out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        self.conv = ConvBlock(in_c, out_c)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class CCUNet(nn.Module):
    def __init__(self, latent_dim: int, num_modalities: int = 4, mod_embed_dim: int = 16):
        super().__init__()
        self.src_embed = nn.Embedding(num_modalities, mod_embed_dim)
        self.tgt_embed = nn.Embedding(num_modalities, mod_embed_dim)
        self.latent_adapter = nn.Linear(latent_dim, 16 * 16)
        self.mod_adapter = nn.Linear(mod_embed_dim * 2, 16 * 16)

        self.inc = ConvBlock(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.outc = nn.Sequential(nn.Conv2d(64, 1, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor, z: torch.Tensor, src_mod_id: torch.Tensor, tgt_mod_id: torch.Tensor) -> torch.Tensor:
        bz, _, h, w = x.shape
        zmap = self.latent_adapter(z).view(bz, 1, 16, 16)
        zmap = F.interpolate(zmap, size=(h, w), mode="bilinear", align_corners=False)

        src_e = self.src_embed(src_mod_id)
        tgt_e = self.tgt_embed(tgt_mod_id)
        mmap = self.mod_adapter(torch.cat([src_e, tgt_e], dim=1)).view(bz, 1, 16, 16)
        mmap = F.interpolate(mmap, size=(h, w), mode="bilinear", align_corners=False)

        x = torch.cat([x, zmap, mmap], dim=1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)
