from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from scipy.linalg import sqrtm

from src.datasets.latent_class_dataset import load_mrm_latent_artifact
from src.utils.latent_contracts import load_latent_payload


def load_latents_file(path: str | Path) -> torch.Tensor:
    payload = load_latent_payload(path)
    return payload["latents"].float().cpu()


def frechet_distance(x: np.ndarray, y: np.ndarray) -> float:
    mu1 = x.mean(axis=0)
    mu2 = y.mean(axis=0)
    sigma1 = np.cov(x, rowvar=False)
    sigma2 = np.cov(y, rowvar=False)
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(np.real(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean)))


def linear_mmd2(x: np.ndarray, y: np.ndarray) -> float:
    dx = x.mean(axis=0)
    dy = y.mean(axis=0)
    return float(np.sum((dx - dy) ** 2))


def compute_fd_metrics(real_artifact_path: str, generated_root: str, modality_names: List[str], max_samples: int) -> Dict[str, Dict[str, float]]:
    real = load_mrm_latent_artifact(real_artifact_path)
    out: Dict[str, Dict[str, float]] = {}
    fd_vals = []
    for cls, name in enumerate(modality_names):
        gen_path = Path(generated_root) / f"generated_latents_class_{cls}.pt"
        gen = load_latents_file(gen_path).numpy()
        ref = real.latents[real.labels == cls][:max_samples].numpy()
        n = min(len(ref), len(gen), max_samples)
        ref = ref[:n]
        gen = gen[:n]
        fd = frechet_distance(ref, gen)
        mmd = linear_mmd2(ref, gen)
        out[name] = {"fd": fd, "mmd": mmd, "num_samples": int(n)}
        fd_vals.append(fd)
    out["overall"] = {"fd_mean": float(np.mean(fd_vals)) if fd_vals else float("nan")}
    return out
