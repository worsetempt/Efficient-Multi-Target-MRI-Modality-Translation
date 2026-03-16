from __future__ import annotations

import math

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim


def reconstruction_metrics(recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict:
    r = ((recon.detach().cpu().numpy() + 1) / 2).clip(0, 1)
    t = ((target.detach().cpu().numpy() + 1) / 2).clip(0, 1)
    m = (mask.detach().cpu().numpy() > 0.5).astype(np.float32)
    c = (1.0 - m) * t + m * r

    full_ssim = sk_ssim(r[0, 0], t[0, 0], data_range=1.0)
    full_psnr = sk_psnr(t[0, 0], r[0, 0], data_range=1.0)
    composite_ssim = sk_ssim(c[0, 0], t[0, 0], data_range=1.0)
    composite_psnr = sk_psnr(t[0, 0], c[0, 0], data_range=1.0)

    denom = float(m.sum())
    if denom < 1:
        mse_masked = 0.0
        psnr_masked = float("inf")
    else:
        mse_masked = float(np.sum(((r - t) ** 2) * m) / denom)
        psnr_masked = float("inf") if mse_masked <= 0 else float(10.0 * np.log10(1.0 / mse_masked))

    return {
        "full_ssim": float(full_ssim),
        "full_psnr": float(full_psnr),
        "composite_ssim": float(composite_ssim),
        "composite_psnr": float(composite_psnr),
        "mse_masked": float(mse_masked),
        "psnr_masked": float(psnr_masked),
    }


@torch.no_grad()
def compute_reconstruction_quality(model, loader, device, max_batches: int = 50) -> dict:
    model.eval()
    stats = {k: [] for k in ["full_ssim", "full_psnr", "composite_ssim", "composite_psnr", "mse_masked", "psnr_masked"]}
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        masked, target, mask, _, _ = batch
        masked = masked.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        recon = model(masked)
        for b in range(recon.shape[0]):
            row = reconstruction_metrics(recon[b:b+1], target[b:b+1], mask[b:b+1])
            for k, v in row.items():
                stats[k].append(v)
    return {f"{k}_mean": float(np.mean(v)) for k, v in stats.items()} | {"n_samples": len(stats["full_ssim"])}
