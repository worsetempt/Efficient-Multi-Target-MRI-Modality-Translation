from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from src.utils.latent_contracts import load_latent_payload


@dataclass
class MultiModalityLatentStore:
    modality_names: List[str]
    latents: Dict[str, torch.Tensor]
    meta: Dict[str, dict]
    length: int
    latent_dim: int

    @classmethod
    def from_paths(cls, latent_paths: Dict[str, str], modality_names: List[str]) -> "MultiModalityLatentStore":
        latents: Dict[str, torch.Tensor] = {}
        meta: Dict[str, dict] = {}
        length: Optional[int] = None
        latent_dim: Optional[int] = None
        reference_ids: Optional[List[str]] = None

        for modality in modality_names:
            payload = load_latent_payload(latent_paths[modality])
            z = payload["latents"].float()
            if z.ndim > 2:
                z = z.reshape(z.shape[0], -1)
            if z.ndim != 2:
                raise ValueError(f"Latents for modality '{modality}' must be [N, D], got {tuple(z.shape)}")

            current_ids = payload.get("sample_ids")
            if current_ids is not None:
                current_ids = [str(x) for x in current_ids]

            if length is None:
                length = int(z.shape[0])
                latent_dim = int(z.shape[1])
                reference_ids = current_ids
            else:
                if z.shape[0] != length:
                    raise ValueError(f"Length mismatch for modality '{modality}': {z.shape[0]} vs {length}")
                if z.shape[1] != latent_dim:
                    raise ValueError(f"Latent dim mismatch for modality '{modality}': {z.shape[1]} vs {latent_dim}")
                if reference_ids is not None and current_ids is not None and current_ids != reference_ids:
                    raise ValueError(f"Sample ordering mismatch detected for modality '{modality}'")

            latents[modality] = z.contiguous()
            meta[modality] = {k: v for k, v in payload.items() if k != "latents"}

        if length is None or latent_dim is None:
            raise ValueError("No latent files were loaded.")
        return cls(modality_names=modality_names, latents=latents, meta=meta, length=length, latent_dim=latent_dim)

    def get(self, modality: str) -> torch.Tensor:
        return self.latents[modality]

    def get_sample_ids(self, modality: str) -> Optional[List[str]]:
        sample_ids = self.meta[modality].get("sample_ids")
        return [str(x) for x in sample_ids] if sample_ids is not None else None

    def __len__(self) -> int:
        return self.length


class AllPairsLatentDataset(Dataset):
    def __init__(self, store: MultiModalityLatentStore, modality_to_id: Dict[str, int], mean: torch.Tensor, std: torch.Tensor):
        self.store = store
        self.modality_to_id = modality_to_id
        self.modality_names = list(modality_to_id.keys())
        self.mean = mean.float()
        self.std = std.float().clamp_min(1e-6)
        self.pairs: List[Tuple[int, int, int]] = []
        for sample_idx in range(len(store)):
            for src_id in range(len(self.modality_names)):
                for tgt_id in range(len(self.modality_names)):
                    if src_id != tgt_id:
                        self.pairs.append((sample_idx, src_id, tgt_id))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        sample_idx, src_id, tgt_id = self.pairs[idx]
        src_m = self.modality_names[src_id]
        tgt_m = self.modality_names[tgt_id]
        z_src = (self.store.get(src_m)[sample_idx] - self.mean) / self.std
        z_tgt = (self.store.get(tgt_m)[sample_idx] - self.mean) / self.std
        return {
            "z_src": z_src,
            "z_tgt": z_tgt,
            "src_id": torch.tensor(src_id, dtype=torch.long),
            "tgt_id": torch.tensor(tgt_id, dtype=torch.long),
            "sample_idx": torch.tensor(sample_idx, dtype=torch.long),
        }
