from __future__ import annotations

from typing import Any, Dict

import torch


REQUIRED_LATENT_KEY = "latents"


def validate_latent_payload(payload: Dict[str, Any]) -> None:
    if REQUIRED_LATENT_KEY not in payload:
        raise ValueError("Latent artifact must contain key 'latents'.")
    latents = payload["latents"]
    if not torch.is_tensor(latents):
        latents = torch.as_tensor(latents)
    if latents.ndim != 2:
        raise ValueError(f"Latents must have shape [N, D], got {tuple(latents.shape)}.")


from pathlib import Path
from typing import Any


def load_latent_payload(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if torch.is_tensor(payload):
        payload = {"latents": payload}
    elif not isinstance(payload, dict):
        payload = {"latents": torch.as_tensor(payload)}
    validate_latent_payload(payload)
    payload["latents"] = torch.as_tensor(payload["latents"], dtype=torch.float32)
    if payload["latents"].ndim > 2:
        payload["latents"] = payload["latents"].reshape(payload["latents"].shape[0], -1)
    return payload


def save_generated_latent_payload(path: str | Path, *, latents, modality: str, modality_id: int, split: str, source: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "latents": torch.as_tensor(latents, dtype=torch.float32),
        "modality": modality,
        "modality_id": int(modality_id),
        "split": split,
        "source": source,
    }
    validate_latent_payload(payload)
    torch.save(payload, path)
