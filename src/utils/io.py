from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def save_checkpoint(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_latent_artifact_pt(path: str | Path, *, latents, labels, paths, split: str, modality_names) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_ids = [Path(p).stem for p in paths]
    payload = {
        "latents": torch.as_tensor(latents, dtype=torch.float32),
        "labels": torch.as_tensor(labels, dtype=torch.long),
        "paths": list(paths),
        "sample_ids": sample_ids,
        "split": split,
        "modality_names": list(modality_names),
        "source": "mrm",
    }
    torch.save(payload, path)


def save_latent_artifact_npz(path: str | Path, *, latents, labels, paths) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, latents=latents, labels=labels, paths=np.array(paths, dtype=object))
