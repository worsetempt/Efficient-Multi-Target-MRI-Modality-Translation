from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.decoder_metrics import evaluate_decoder
from src.evaluation.decoder_visualization import save_decoder_preview_grid
from src.utils.io import save_checkpoint, save_json


def gradient_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    dx_p = preds[..., :, 1:] - preds[..., :, :-1]
    dx_t = targets[..., :, 1:] - targets[..., :, :-1]
    dy_p = preds[..., 1:, :] - preds[..., :-1, :]
    dy_t = targets[..., 1:, :] - targets[..., :-1, :]
    return (dx_p - dx_t).abs().mean() + (dy_p - dy_t).abs().mean()


def gaussian_window(window_size: int = 11, sigma: float = 1.5, channels: int = 1) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_1d = g.unsqueeze(1)
    window_2d = window_1d @ window_1d.t()
    window_2d = window_2d / window_2d.sum()
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim_loss(preds: torch.Tensor, targets: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    channels = preds.size(1)
    window = gaussian_window(window_size=window_size, channels=channels).to(preds.device, preds.dtype)
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
    return 1.0 - ssim_map.mean()


def train_decoder(model, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, cfg: dict, exp_paths: dict) -> Dict[str, List[float]]:
    tr = cfg["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tr["lr"]), weight_decay=float(tr["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(tr["epochs"]))
    scaler = torch.cuda.amp.GradScaler(enabled=(bool(cfg["runtime"]["amp"]) and device.type == "cuda"))
    use_amp = bool(cfg["runtime"]["amp"]) and device.type == "cuda"

    best_val = float("inf")
    history: Dict[str, List[float]] = {"train_l1": [], "val_l1": [], "val_psnr": [], "val_ssim": []}

    for epoch in range(1, int(tr["epochs"]) + 1):
        model.train()
        total_l1 = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{int(tr['epochs'])}", leave=False)
        for latents, targets, _, _ in pbar:
            latents = latents.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                preds = model(latents)
                loss_l1 = F.l1_loss(preds, targets)
                loss = loss_l1 + float(tr["grad_weight"]) * gradient_loss(preds, targets)
                if float(tr["ssim_weight"]) > 0:
                    loss = loss + float(tr["ssim_weight"]) * ssim_loss(preds, targets)
            scaler.scale(loss).backward()
            if float(tr["grad_clip"]) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(tr["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            bs = latents.size(0)
            total_l1 += float(loss_l1.item()) * bs
            n += bs
            pbar.set_postfix(train_l1=total_l1 / max(n, 1))
        scheduler.step()

        train_l1 = total_l1 / max(n, 1)
        val_metrics = evaluate_decoder(model, val_loader, device, amp=bool(cfg["runtime"]["amp"]))
        history["train_l1"].append(train_l1)
        history["val_l1"].append(val_metrics["l1"])
        history["val_psnr"].append(val_metrics["psnr"])
        history["val_ssim"].append(val_metrics["ssim"])

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "latent_dim": int(cfg["model"]["latent_dim"]),
            "base_ch": int(cfg["model"]["base_ch"]),
            "amp": bool(cfg["runtime"]["amp"]),
            "history": history,
        }
        save_checkpoint(exp_paths["checkpoints"] / "last_decoder.pt", ckpt)
        if val_metrics["l1"] < best_val:
            best_val = val_metrics["l1"]
            save_checkpoint(exp_paths["checkpoints"] / "best_decoder.pt", ckpt)

        if epoch == 1 or epoch % int(tr["preview_every"]) == 0 or epoch == int(tr["epochs"]):
            save_decoder_preview_grid(
                model=model,
                loader=val_loader,
                device=device,
                out_path=exp_paths["visuals"] / f"preview_epoch{epoch}.png",
                amp=bool(cfg["runtime"]["amp"]),
                max_items=int(tr["preview_items"]),
            )

    save_json(exp_paths["metrics"] / "history.json", history)
    return history
