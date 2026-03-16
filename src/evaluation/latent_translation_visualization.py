from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from src.utils.latent_contracts import load_latent_payload


def plot_generated_umap(real_root_by_modality: Dict[str, str], generated_root: str | Path, output_path: str | Path, max_samples: int = 1500, n_neighbors: int = 15, min_dist: float = 0.1) -> str:
    try:
        import umap  # type: ignore
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric="euclidean", random_state=42)
        reducer_name = "umap"
    except Exception:
        reducer = PCA(n_components=2, random_state=42)
        reducer_name = "pca_fallback"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated_root = Path(generated_root)
    all_x = []
    all_y = []
    label_names = []

    for modality, path in real_root_by_modality.items():
        x = load_latent_payload(path)["latents"][:max_samples].cpu().numpy()
        all_x.append(x)
        all_y.append(np.full(x.shape[0], len(label_names)))
        label_names.append(f"real_{modality}")

    for gen_file in sorted(generated_root.glob("*_to_*.pt")):
        x = load_latent_payload(gen_file)["latents"][:max_samples].cpu().numpy()
        all_x.append(x)
        all_y.append(np.full(x.shape[0], len(label_names)))
        label_names.append(gen_file.stem)

    X = np.concatenate(all_x, axis=0)
    y = np.concatenate(all_y, axis=0)
    emb = reducer.fit_transform(X)
    plt.figure(figsize=(10, 8))
    for label_idx, name in enumerate(label_names):
        mask = y == label_idx
        plt.scatter(emb[mask, 0], emb[mask, 1], s=8, alpha=0.65, label=name)
    plt.legend(markerscale=2, fontsize=7)
    plt.title(f"Latent {reducer_name.upper()} projection")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return str(output_path)
