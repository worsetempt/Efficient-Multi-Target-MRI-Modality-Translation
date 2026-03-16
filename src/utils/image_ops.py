from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

IMG_SIZE = 224


def build_grayscale_transform(img_size: int = IMG_SIZE):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def denorm_to_uint8(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().clamp(-1.0, 1.0)
    x = ((x + 1.0) / 2.0 * 255.0).round().to(torch.uint8)
    return x.squeeze(0).numpy()


def load_grayscale_image(path: str, img_size: int = IMG_SIZE) -> torch.Tensor:
    img = Image.open(path).convert("L")
    return build_grayscale_transform(img_size)(img)
