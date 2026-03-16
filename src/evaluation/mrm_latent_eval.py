from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def maybe_silhouette(latents: np.ndarray, labels: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2 or len(latents) < 10:
        return None
    x = StandardScaler().fit_transform(latents.astype(np.float32))
    return float(silhouette_score(x, labels))


def maybe_linear_probe(latents: np.ndarray, labels: np.ndarray, cv: int = 3):
    if len(np.unique(labels)) < 2 or len(latents) < 10:
        return None
    x = StandardScaler().fit_transform(latents.astype(np.float32))
    clf = LogisticRegression(max_iter=500, random_state=42)
    n_splits = min(cv, len(np.unique(labels)))
    scores = cross_val_score(clf, x, labels, cv=n_splits, scoring="accuracy")
    return {"mean": float(np.mean(scores)), "std": float(np.std(scores)), "cv": int(n_splits)}
