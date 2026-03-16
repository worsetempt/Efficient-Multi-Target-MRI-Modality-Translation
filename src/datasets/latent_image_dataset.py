from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.utils.image_ops import build_grayscale_transform
from src.utils.latent_contracts import load_latent_payload


class LatentImageDataset(Dataset):
    def __init__(self, npz_path: str, img_size: int = 224):
        data = np.load(npz_path, allow_pickle=True)
        self.latents = torch.as_tensor(data["latents"], dtype=torch.float32)
        self.labels = torch.as_tensor(data["labels"], dtype=torch.long)
        self.paths = [str(p) for p in data["paths"]]
        if self.latents.ndim != 2:
            self.latents = self.latents.view(self.latents.size(0), -1)
        self.transform = build_grayscale_transform(img_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("L")
        target = self.transform(img)
        return self.latents[idx], target, self.labels[idx], self.paths[idx]


class GeneratedLatentEvalDataset(Dataset):
    def __init__(self, latent_path: str, reference_npz: Optional[str] = None, img_size: int = 224):
        payload = load_latent_payload(latent_path)
        self.latents = payload["latents"].float()
        self.paths: List[str] = [str(p) for p in payload.get("paths", [])]
        self.sample_ids: List[str] = [str(x) for x in payload.get("sample_ids", [])]
        self.labels = torch.as_tensor(payload.get("labels", torch.zeros(len(self.latents))), dtype=torch.long)
        if self.latents.ndim != 2:
            self.latents = self.latents.view(self.latents.size(0), -1)
        self.transform = build_grayscale_transform(img_size)
        self.reference_by_id = None
        self.reference_by_path = None
        if reference_npz:
            data = np.load(reference_npz, allow_pickle=True)
            ref_paths = [str(p) for p in data["paths"]]
            ref_sample_ids = [Path(p).stem for p in ref_paths]
            self.reference_by_id = {sid: p for sid, p in zip(ref_sample_ids, ref_paths)}
            self.reference_by_path = {p: p for p in ref_paths}

    def __len__(self) -> int:
        return self.latents.shape[0]

    def _resolve_target_path(self, idx: int) -> Optional[str]:
        if idx < len(self.paths) and self.paths[idx]:
            return self.paths[idx]
        if self.reference_by_id is not None and idx < len(self.sample_ids):
            sid = self.sample_ids[idx]
            if sid in self.reference_by_id:
                return self.reference_by_id[sid]
        return None

    def __getitem__(self, idx: int):
        target_path = self._resolve_target_path(idx)
        target = None
        if target_path is not None:
            img = Image.open(target_path).convert("L")
            target = self.transform(img)
        return self.latents[idx], target, self.labels[idx], target_path or ""
