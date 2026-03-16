from __future__ import annotations

import math
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity



def batch_l1(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(pred - tgt)).item())



def batch_psnr(pred: torch.Tensor, tgt: torch.Tensor, max_val: float = 1.0) -> float:
    mse = float(torch.mean((pred - tgt) ** 2).item())
    if mse == 0.0:
        return float("inf")
    return float(20.0 * math.log10(max_val) - 10.0 * math.log10(mse))



def batch_ssim(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    pred = pred.detach().cpu().clamp(0, 1)
    tgt = tgt.detach().cpu().clamp(0, 1)
    vals = []
    for i in range(pred.size(0)):
        p = pred[i].squeeze(0).numpy()
        t = tgt[i].squeeze(0).numpy()
        vals.append(structural_similarity(t, p, data_range=1.0))
    return float(sum(vals) / max(len(vals), 1))



def single_image_metrics(pred: torch.Tensor, tgt: torch.Tensor) -> Dict[str, float]:
    pred_np = pred.detach().cpu().clamp(0, 1).squeeze().numpy()
    tgt_np = tgt.detach().cpu().clamp(0, 1).squeeze().numpy()
    return {
        "l1": float(F.l1_loss(pred, tgt).item()),
        "psnr": float(peak_signal_noise_ratio(tgt_np, pred_np, data_range=1.0)),
        "ssim": float(structural_similarity(tgt_np, pred_np, data_range=1.0)),
    }



def summarize_metric_list(metrics: list[Dict[str, float]]) -> Dict[str, float]:
    if not metrics:
        return {"average_l1": float("nan"), "average_psnr": float("nan"), "average_ssim": float("nan"), "num_samples": 0}
    return {
        "average_l1": float(np.mean([m["l1"] for m in metrics])),
        "average_psnr": float(np.mean([m["psnr"] for m in metrics])),
        "average_ssim": float(np.mean([m["ssim"] for m in metrics])),
        "num_samples": int(len(metrics)),
    }
