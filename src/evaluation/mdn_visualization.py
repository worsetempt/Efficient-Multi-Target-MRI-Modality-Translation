from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from src.datasets.latent_class_dataset import load_mrm_latent_artifact
from src.evaluation.mdn_metrics import load_latents_file


def plot_generated_umap(real_artifact_path: str, generated_root: str, modality_names: List[str], output_path: str | Path, max_samples: int, n_neighbors: int = 15, min_dist: float = 0.1) -> str:
    try:
        import umap  # type: ignore
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric="euclidean", random_state=42)
        name = "UMAP"
    except Exception:
        reducer = PCA(n_components=2, random_state=42)
        name = "PCA"

    real = load_mrm_latent_artifact(real_artifact_path)
    Xs = []
    ys = []
    labels = []
    for cls, mod in enumerate(modality_names):
        rx = real.latents[real.labels == cls][:max_samples].numpy()
        gx = load_latents_file(Path(generated_root) / f"generated_latents_class_{cls}.pt")[:max_samples].numpy()
        Xs.extend([rx, gx])
        ys.extend([np.full(len(rx), cls), np.full(len(gx), cls + len(modality_names))])
        labels.extend([f"real:{mod}", f"gen:{mod}"])
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    emb = reducer.fit_transform(X)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 7))
    for label_val, label_name in zip(np.unique(y), labels):
        idx = y == label_val
        plt.scatter(emb[idx, 0], emb[idx, 1], s=8, alpha=0.65, label=label_name)
    plt.legend(markerscale=2, fontsize=8)
    plt.title(f"MDN Generated Latents ({name})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return str(output_path)
