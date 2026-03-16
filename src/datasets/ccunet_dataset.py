from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms

from src.utils.latent_contracts import load_latent_payload


@dataclass
class GeneratedLatentBatch:
    latents: torch.Tensor
    source_modality: Optional[str]
    target_modality: Optional[str]


@dataclass
class MRMAlignmentIndex:
    latent_by_sample_id: Dict[str, torch.Tensor]
    modality_names: List[str]
    latent_dim: int



def build_image_transform(image_size: int):
    return transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])



def parse_flat_braTS_filename(fname: str) -> Optional[Tuple[str, str, int]]:
    stem = Path(fname).stem
    parts = stem.split("_")
    if len(parts) < 5 or parts[-2] != "slice":
        return None
    try:
        slice_idx = int(parts[-1])
    except ValueError:
        return None
    return f"{parts[0]}_{parts[1]}", parts[2], slice_idx



def build_mrm_alignment_index(mrm_latent_path: str, modality_names: Sequence[str]) -> MRMAlignmentIndex:
    payload = load_latent_payload(mrm_latent_path)
    latents = payload["latents"].float()
    labels = payload.get("labels")
    paths = payload.get("paths")
    artifact_modalities = list(payload.get("modality_names", []))
    if labels is None or paths is None:
        raise ValueError("MRM artifact must contain 'labels' and 'paths' for CCU-Net training.")
    if artifact_modalities and artifact_modalities != list(modality_names):
        raise ValueError("Configured modality_names do not match MRM artifact modality_names.")
    labels = torch.as_tensor(labels, dtype=torch.long)
    if len(paths) != latents.shape[0] or labels.shape[0] != latents.shape[0]:
        raise ValueError("MRM artifact lengths for latents, labels, and paths must match.")

    latent_by_sample_id: Dict[str, torch.Tensor] = {}
    for z, label, path in zip(latents, labels.tolist(), paths):
        parsed = parse_flat_braTS_filename(os.path.basename(str(path)))
        if parsed is None:
            continue
        case_id, modality, slice_idx = parsed
        expected = list(modality_names)[label]
        if modality != expected:
            continue
        sample_id = f"{case_id}_{modality}_slice_{slice_idx}"
        latent_by_sample_id[sample_id] = z.float().clone()

    if not latent_by_sample_id:
        raise ValueError("No valid aligned MRM latent entries could be built from artifact.")
    latent_dim = int(next(iter(latent_by_sample_id.values())).numel())
    return MRMAlignmentIndex(latent_by_sample_id=latent_by_sample_id, modality_names=list(modality_names), latent_dim=latent_dim)


class AllPairsMRMDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        mrm_latent_path: str,
        modalities: Sequence[str],
        image_size: int = 128,
        start_index: int = 0,
        max_aligned_slices: Optional[int] = None,
    ):
        self.root_dir = root_dir
        self.modalities = list(modalities)
        self.mod_to_id = {m: i for i, m in enumerate(self.modalities)}
        self.transform = build_image_transform(image_size)
        self.mrm_index = build_mrm_alignment_index(mrm_latent_path, self.modalities)

        grouped: Dict[Tuple[str, int], Dict[str, str]] = {}
        for fname in sorted(os.listdir(root_dir)):
            if not fname.lower().endswith(".png"):
                continue
            parsed = parse_flat_braTS_filename(fname)
            if parsed is None:
                continue
            case_id, modality, slice_idx = parsed
            if modality not in self.mod_to_id:
                continue
            grouped.setdefault((case_id, slice_idx), {})[modality] = os.path.join(root_dir, fname)

        aligned_keys: List[Tuple[str, int]] = []
        for key, mod_dict in grouped.items():
            if all(m in mod_dict for m in self.modalities):
                ok = True
                for m in self.modalities:
                    sample_id = f"{key[0]}_{m}_slice_{key[1]}"
                    if sample_id not in self.mrm_index.latent_by_sample_id:
                        ok = False
                        break
                if ok:
                    aligned_keys.append(key)
        aligned_keys = sorted(aligned_keys)
        end_index = None if max_aligned_slices is None else start_index + max_aligned_slices
        aligned_keys = aligned_keys[start_index:end_index]
        if not aligned_keys:
            raise ValueError("No fully aligned slices found with matching MRM latents.")

        self.images_by_mod = {m: [] for m in self.modalities}
        for key in aligned_keys:
            mod_dict = grouped[key]
            for m in self.modalities:
                self.images_by_mod[m].append(mod_dict[m])

        self.samples: List[Tuple[int, str, str]] = []
        for idx in range(len(aligned_keys)):
            for src_mod in self.modalities:
                for tgt_mod in self.modalities:
                    if src_mod != tgt_mod:
                        self.samples.append((idx, src_mod, tgt_mod))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        idx, src_mod, tgt_mod = self.samples[index]
        src_path = self.images_by_mod[src_mod][idx]
        tgt_path = self.images_by_mod[tgt_mod][idx]
        src = self.transform(Image.open(src_path).convert("L"))
        tgt = self.transform(Image.open(tgt_path).convert("L"))

        parsed = parse_flat_braTS_filename(os.path.basename(tgt_path))
        assert parsed is not None
        case_id, _, slice_idx = parsed
        sample_id = f"{case_id}_{tgt_mod}_slice_{slice_idx}"
        tgt_latent = self.mrm_index.latent_by_sample_id[sample_id]

        return {
            "src": src,
            "tgt": tgt,
            "tgt_latent": tgt_latent,
            "src_mod_id": torch.tensor(self.mod_to_id[src_mod], dtype=torch.long),
            "tgt_mod_id": torch.tensor(self.mod_to_id[tgt_mod], dtype=torch.long),
            "src_path": src_path,
            "tgt_path": tgt_path,
            "src_modality": src_mod,
            "tgt_modality": tgt_mod,
        }



def make_sample_subsets(dataset: AllPairsMRMDataset, train_per_target: int = 3000, val_total: int = 2000, seed: int = 42):
    import random
    rng = random.Random(seed)
    by_tgt = {m: [] for m in dataset.modalities}
    for i, (_, _, tgt_mod) in enumerate(dataset.samples):
        by_tgt[tgt_mod].append(i)
    for m in dataset.modalities:
        rng.shuffle(by_tgt[m])
    train_indices: List[int] = []
    remaining: List[int] = []
    for m in dataset.modalities:
        bucket = by_tgt[m]
        if len(bucket) < train_per_target:
            raise ValueError(f"Not enough samples for target modality {m}: have {len(bucket)}, need {train_per_target}.")
        train_indices.extend(bucket[:train_per_target])
        remaining.extend(bucket[train_per_target:])
    rng.shuffle(remaining)
    if len(remaining) < val_total:
        raise ValueError(f"Not enough remaining samples for val: have {len(remaining)}, need {val_total}.")
    val_indices = remaining[:val_total]
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return Subset(dataset, train_indices), Subset(dataset, val_indices)



def load_generated_latent_batch(path: str) -> GeneratedLatentBatch:
    payload = load_latent_payload(path)
    return GeneratedLatentBatch(
        latents=payload["latents"].float(),
        source_modality=payload.get("source_modality"),
        target_modality=payload.get("target_modality") or payload.get("modality"),
    )
