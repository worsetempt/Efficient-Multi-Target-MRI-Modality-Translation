from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from src.utils.image_ops import IMG_SIZE, denorm_to_uint8
from src.utils.plotting import save_stacked_rows


@torch.no_grad()
def save_decoder_preview_grid(model, loader, device: torch.device, out_path: str | Path, amp: bool, max_items: int = 8) -> str:
    model.eval()
    rows: List[np.ndarray] = []
    use_amp = amp and device.type == "cuda"
    count = 0
    for latents, targets, _, _ in loader:
        latents = latents.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            preds = model(latents)
        for i in range(preds.size(0)):
            gt = denorm_to_uint8(targets[i])
            pr = denorm_to_uint8(preds[i])
            gap = np.full((IMG_SIZE, 8), 255, dtype=np.uint8)
            rows.append(np.concatenate([gt, gap, pr], axis=1))
            count += 1
            if count >= max_items:
                return save_stacked_rows(rows, out_path)
    return save_stacked_rows(rows, out_path)


@torch.no_grad()
def save_decoded_outputs(model, loader, device: torch.device, out_dir: str | Path, amp: bool, max_items: Optional[int] = None) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    use_amp = amp and device.type == "cuda"
    written = 0
    for latents, _, _, _ in loader:
        latents = latents.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            preds = model(latents)
        for i in range(preds.size(0)):
            arr = denorm_to_uint8(preds[i])
            Image.fromarray(arr, mode="L").save(out_dir / f"decoded_{written:05d}.png")
            written += 1
            if max_items is not None and written >= max_items:
                return str(out_dir)
    return str(out_dir)
