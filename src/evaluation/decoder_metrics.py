from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.utils.metrics_common import psnr_from_mse


def _gaussian_window(window_size: int = 11, sigma: float = 1.5, channels: int = 1) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_1d = g.unsqueeze(1)
    window_2d = window_1d @ window_1d.t()
    window_2d = window_2d / window_2d.sum()
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim_score(preds: torch.Tensor, targets: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    channels = preds.size(1)
    window = _gaussian_window(window_size=window_size, channels=channels).to(preds.device, preds.dtype)
    mu_x = F.conv2d(preds, window, padding=window_size // 2, groups=channels)
    mu_y = F.conv2d(targets, window, padding=window_size // 2, groups=channels)
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(preds * preds, window, padding=window_size // 2, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(targets * targets, window, padding=window_size // 2, groups=channels) - mu_y2
    sigma_xy = F.conv2d(preds * targets, window, padding=window_size // 2, groups=channels) - mu_xy
    c1 = 0.0004
    c2 = 0.0036
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-8)
    return ssim_map.mean()


def evaluate_decoder(model, loader, device: torch.device, amp: bool) -> Dict[str, float]:
    model.eval()
    total_l1 = 0.0
    total_mse = 0.0
    total_ssim = 0.0
    n = 0
    use_amp = amp and device.type == "cuda"
    with torch.no_grad():
        for latents, targets, _, _ in tqdm(loader, desc="Val", leave=False):
            latents = latents.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                preds = model(latents)
                l1 = F.l1_loss(preds, targets, reduction="mean")
                mse = F.mse_loss(preds, targets, reduction="mean")
                ssim = ssim_score(preds, targets)
            bs = latents.size(0)
            total_l1 += float(l1.item()) * bs
            total_mse += float(mse.item()) * bs
            total_ssim += float(ssim.item()) * bs
            n += bs
    mse_mean = total_mse / max(n, 1)
    return {
        "l1": total_l1 / max(n, 1),
        "mse": mse_mean,
        "psnr": psnr_from_mse(mse_mean, data_range=2.0),
        "ssim": total_ssim / max(n, 1),
    }
