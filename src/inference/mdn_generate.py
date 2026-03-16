from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch

from src.utils.io import save_json
from src.utils.latent_contracts import save_generated_latent_payload


def save_generated_outputs(outputs: Dict[str, torch.Tensor], modality_names: List[str], output_dir: str | Path, split: str = "generated") -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    for cls_str, latents in outputs.items():
        cls = int(cls_str)
        path = output_dir / f"generated_latents_class_{cls}.pt"
        save_generated_latent_payload(path, latents=latents, modality=modality_names[cls], modality_id=cls, split=split, source="mdn")
        saved[modality_names[cls]] = str(path)
    save_json(output_dir / "generation_summary.json", {"files": saved, "source": "mdn"})
    return saved
