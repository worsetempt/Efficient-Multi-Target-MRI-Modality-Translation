from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.utils.io import save_latent_artifact_npz, save_latent_artifact_pt


@torch.no_grad()
def extract_latents(model, loader, device, max_samples: int = 0):
    model.eval()
    latents, labels, paths = [], [], []
    for batch in tqdm(loader, desc="Extracting latents", leave=False):
        masked, _, _, mod_ids, path_list = batch
        masked = masked.to(device, non_blocking=True)
        z = model.encode(masked)
        latents.append(z.detach().cpu().numpy())
        labels.append(mod_ids.detach().cpu().numpy())
        paths.extend(path_list)
        if max_samples > 0 and len(paths) >= max_samples:
            break
    lat = np.concatenate(latents, axis=0) if latents else np.empty((0, 0), dtype=np.float32)
    lab = np.concatenate(labels, axis=0) if labels else np.empty((0,), dtype=np.int64)
    if max_samples > 0:
        lat = lat[:max_samples]
        lab = lab[:max_samples]
        paths = paths[:max_samples]
    return lat, lab, paths


def extract_all_splits(model, loaders: dict, device, output_dir: str | Path, modality_names):
    output_dir = Path(output_dir)
    summary = {}
    for split, loader in loaders.items():
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        lat, lab, paths = extract_latents(model, loader, device)
        save_latent_artifact_pt(split_dir / f"latents_{split}.pt", latents=lat, labels=lab, paths=paths, split=split, modality_names=modality_names)
        save_latent_artifact_npz(split_dir / f"latents_{split}.npz", latents=lat, labels=lab, paths=paths)
        summary[split] = {"num_samples": int(len(paths)), "latent_dim": int(lat.shape[1]) if lat.size else 0}
    return summary
