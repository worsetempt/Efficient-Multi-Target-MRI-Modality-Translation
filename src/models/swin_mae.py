from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mrm_blocks import SkipDecoder

try:
    from timm import create_model
except ImportError:
    create_model = None


class SwinMAE(nn.Module):
    def __init__(self, architecture: str, in_chans: int = 1, pretrained: bool = True, num_classes: int = 0):
        super().__init__()
        if create_model is None:
            raise ImportError("timm required. Install: pip install timm")
        self.encoder = create_model(
            architecture,
            pretrained=pretrained,
            in_chans=in_chans,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        chs = self.encoder.feature_info.channels()
        self.chs = list(chs)
        self.encoder_dim = chs[-1]
        self.decoder = SkipDecoder(chs, out_ch=in_chans)
        self.num_classes = num_classes
        self.aux_head = nn.Linear(self.encoder_dim, num_classes) if num_classes > 0 else None

    def _ensure_nchw(self, f: torch.Tensor) -> torch.Tensor:
        if f.ndim == 4 and (f.shape[-1] in self.chs) and (f.shape[1] not in self.chs):
            return f.permute(0, 3, 1, 2).contiguous()
        return f

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        f4 = self._ensure_nchw(feats[-1])
        return F.adaptive_avg_pool2d(f4, 1).flatten(1)

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        feats = self.encoder(x)
        f1, f2, f3, f4 = [self._ensure_nchw(f) for f in feats]
        recon = self.decoder(f1, f2, f3, f4)
        if recon.shape[-2:] != x.shape[-2:]:
            recon = F.interpolate(recon, size=x.shape[-2:], mode="nearest")
        if return_latent:
            z = F.adaptive_avg_pool2d(f4, 1).flatten(1)
            return recon, z
        return recon
