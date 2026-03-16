from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

try:
    import umap
except ImportError:
    umap = None


def plot_umap(latents, labels, modality_names, save_path, title="UMAP of MRM Latents", n_neighbors: int = 15, min_dist: float = 0.1):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if umap is None:
        return None
    x = StandardScaler().fit_transform(latents.astype(np.float32))
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=42)
    emb = reducer.fit_transform(x)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for mid in np.unique(labels):
        idx = labels == mid
        name = modality_names[mid] if mid < len(modality_names) else f"class_{mid}"
        ax.scatter(emb[idx, 0], emb[idx, 1], s=8, alpha=0.7, c=colors[mid % len(colors)], label=name)
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(loc="best", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(save_path)


def save_reconstruction_examples(model, loader, device, save_dir, n: int = 8):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    def denorm(x):
        return (x * 0.5 + 0.5).clamp(0.0, 1.0)

    model.eval()
    batch = next(iter(loader))
    masked, target, mask, _, _ = batch
    masked = masked.to(device)
    target = target.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        recon = model(masked)
    composite = (1.0 - mask) * target + mask * recon

    masked_v = denorm(masked).cpu()
    target_v = denorm(target).cpu()
    recon_v = denorm(recon).cpu()
    comp_v = denorm(composite).cpu()
    mask_v = mask.cpu()

    for i in range(min(n, masked_v.shape[0])):
        fig, ax = plt.subplots(1, 5, figsize=(15, 3))
        ax[0].imshow(masked_v[i, 0], cmap="gray", vmin=0, vmax=1)
        ax[0].set_title("Masked Input")
        ax[1].imshow(recon_v[i, 0], cmap="gray", vmin=0, vmax=1)
        ax[1].set_title("Reconstruction")
        ax[2].imshow(comp_v[i, 0], cmap="gray", vmin=0, vmax=1)
        ax[2].set_title("Composite")
        ax[3].imshow(target_v[i, 0], cmap="gray", vmin=0, vmax=1)
        ax[3].set_title("Ground Truth")
        ax[4].imshow(mask_v[i, 0], cmap="gray", vmin=0, vmax=1)
        ax[4].set_title("Mask")
        for a in ax:
            a.axis("off")
        plt.tight_layout()
        plt.savefig(save_dir / f"recon_{i}.png", dpi=200, bbox_inches="tight")
        plt.close()
