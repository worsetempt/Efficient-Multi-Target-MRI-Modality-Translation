from __future__ import annotations

import random
import re
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


MODALITIES = ["t1", "t2", "t1ce", "flair"]
FILE_RE = re.compile(r"^BraTS2021_(\d+)_(t1|t2|t1ce|flair)_slice_(\d+)\.png$")
IMG_SIZE = 224


class BraTSMAEDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        modalities=None,
        mask_ratio: float = 0.25,
        patch_size: int = 16,
        max_per_modality: int = 0,
        seed: int = 42,
        img_size: int = IMG_SIZE,
    ):
        self.data_dir = Path(data_dir)
        self.mask_ratio = float(mask_ratio)
        self.patch_size = int(patch_size)
        self.modalities = modalities or MODALITIES
        self.mod_to_id = {m: i for i, m in enumerate(self.modalities)}

        samples = []
        for fn in sorted(self.data_dir.iterdir()):
            m = FILE_RE.match(fn.name)
            if m and m.group(2).lower() in self.mod_to_id:
                samples.append((str(fn), self.mod_to_id[m.group(2).lower()]))

        if max_per_modality and max_per_modality > 0:
            rng = random.Random(seed)
            by_mod = {}
            for path, mid in samples:
                by_mod.setdefault(mid, []).append((path, mid))
            samples = []
            for mid in sorted(by_mod.keys()):
                lst = by_mod[mid]
                rng.shuffle(lst)
                samples.extend(lst[:max_per_modality])

        self.samples = samples
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def _make_mask(self, h: int, w: int) -> torch.Tensor:
        ph, pw = h // self.patch_size, w // self.patch_size
        n = ph * pw
        k = int(n * self.mask_ratio)
        patch_mask = torch.zeros(n, dtype=torch.bool)
        idx = torch.randperm(n)[:k]
        patch_mask[idx] = True
        patch_mask = patch_mask.view(ph, pw)
        mask = patch_mask.repeat_interleave(self.patch_size, 0).repeat_interleave(self.patch_size, 1)
        return mask.unsqueeze(0).float()

    def __getitem__(self, idx):
        path, mod_id = self.samples[idx]
        img = Image.open(path).convert("L")
        x = self.transform(img)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        h, w = x.shape[-2:]
        mask = self._make_mask(h, w)
        masked_x = x * (1 - mask)
        return masked_x, x, mask, mod_id, path
