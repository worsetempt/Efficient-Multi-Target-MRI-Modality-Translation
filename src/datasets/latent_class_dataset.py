from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, Sampler

from src.utils.latent_contracts import load_latent_payload


@dataclass
class LatentArtifactView:
    latents: torch.Tensor
    labels: torch.Tensor
    modality_names: List[str]


class SingleClassLatentDataset(Dataset):
    def __init__(self, latents: torch.Tensor, class_label: int):
        self.latents = latents.float().contiguous()
        self.class_label = int(class_label)

    def __len__(self) -> int:
        return int(self.latents.shape[0])

    def __getitem__(self, idx: int):
        return self.latents[idx], torch.tensor(self.class_label, dtype=torch.long)


class NormalizedLatentDataset(Dataset):
    def __init__(self, base_dataset: ConcatDataset, mean: torch.Tensor, std: torch.Tensor):
        self.base_dataset = base_dataset
        self.mean = mean.float()
        self.std = std.float()

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        x, y = self.base_dataset[idx]
        return (x - self.mean) / self.std, y


class BalancedClassBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset: ConcatDataset, batch_size: int, num_classes: int, drop_last: bool = True):
        if batch_size % num_classes != 0:
            raise ValueError(f"batch_size={batch_size} must be divisible by num_classes={num_classes}")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_classes = int(num_classes)
        self.per_class = self.batch_size // self.num_classes
        self.drop_last = drop_last

        self.class_indices: Dict[int, List[int]] = {c: [] for c in range(self.num_classes)}
        offset = 0
        for subset in self.dataset.datasets:
            for i in range(len(subset)):
                _, label = subset[i]
                self.class_indices[int(label)].append(offset + i)
            offset += len(subset)

        self.min_class_count = min(len(v) for v in self.class_indices.values())
        self.num_batches = self.min_class_count // self.per_class if drop_last else int(np.ceil(self.min_class_count / self.per_class))

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        shuffled = {c: np.random.permutation(v).tolist() for c, v in self.class_indices.items()}
        for b in range(self.num_batches):
            batch = []
            for c in range(self.num_classes):
                start = b * self.per_class
                end = start + self.per_class
                chunk = shuffled[c][start:end]
                if len(chunk) < self.per_class:
                    if self.drop_last:
                        return
                    reps = (self.per_class + max(1, len(chunk)) - 1) // max(1, len(chunk))
                    chunk = (chunk * reps)[: self.per_class]
                batch.extend(chunk)
            np.random.shuffle(batch)
            yield batch


def load_mrm_latent_artifact(path: str) -> LatentArtifactView:
    payload = load_latent_payload(path)
    latents = payload["latents"]
    labels = payload.get("labels")
    modality_names = payload.get("modality_names")
    if labels is None:
        raise ValueError("MRM latent artifact must contain 'labels' for MDN training.")
    if modality_names is None:
        raise ValueError("MRM latent artifact must contain 'modality_names' for MDN training.")
    labels = torch.as_tensor(labels, dtype=torch.long)
    if latents.shape[0] != labels.shape[0]:
        raise ValueError("Latents and labels length mismatch in MRM artifact.")
    return LatentArtifactView(latents=latents.float(), labels=labels, modality_names=list(modality_names))


def split_by_class(view: LatentArtifactView, modality_names: List[str]) -> List[SingleClassLatentDataset]:
    if list(view.modality_names) != list(modality_names):
        raise ValueError("Configured modality_names do not match latent artifact modality_names.")
    datasets: List[SingleClassLatentDataset] = []
    for class_idx, _ in enumerate(modality_names):
        mask = view.labels == class_idx
        cls_latents = view.latents[mask]
        if cls_latents.numel() == 0:
            raise ValueError(f"No latents found for class {class_idx} ({modality_names[class_idx]}).")
        datasets.append(SingleClassLatentDataset(cls_latents, class_idx))
    return datasets
