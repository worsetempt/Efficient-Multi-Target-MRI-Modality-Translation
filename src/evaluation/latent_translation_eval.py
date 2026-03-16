from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from scipy.linalg import sqrtm

from src.utils.latent_contracts import load_latent_payload


def frechet_distance(x: np.ndarray, y: np.ndarray) -> float:
    mu1 = x.mean(axis=0)
    mu2 = y.mean(axis=0)
    sigma1 = np.cov(x, rowvar=False)
    sigma2 = np.cov(y, rowvar=False)
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return float(np.real(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean)))


def _get_pair_target_latents(real_root_by_modality: Dict[str, str], target_modality: str, max_samples: int) -> np.ndarray:
    payload = load_latent_payload(real_root_by_modality[target_modality])
    return payload["latents"][:max_samples].cpu().numpy()


def compute_fd_metrics(real_root_by_modality: Dict[str, str], generated_root: str | Path, max_samples: int = 5000) -> Dict[str, float]:
    generated_root = Path(generated_root)
    pair_metrics: Dict[str, float] = {}
    fd_values = []
    for gen_file in sorted(generated_root.glob("*_to_*.pt")):
        payload = load_latent_payload(gen_file)
        target_modality = str(payload["target_modality"])
        gen = payload["latents"][:max_samples].cpu().numpy()
        real = _get_pair_target_latents(real_root_by_modality, target_modality, min(max_samples, len(gen)))
        n = min(len(gen), len(real))
        fd = frechet_distance(real[:n], gen[:n])
        pair_metrics[gen_file.stem] = fd
        fd_values.append(fd)
    pair_metrics["fd_mean"] = float(np.mean(fd_values)) if fd_values else float("nan")
    return pair_metrics
