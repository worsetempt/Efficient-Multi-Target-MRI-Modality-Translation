from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.datasets.ccunet_dataset import build_image_transform, load_generated_latent_batch, parse_flat_braTS_filename
from src.evaluation.ccunet_metrics import single_image_metrics, summarize_metric_list
from src.evaluation.ccunet_visualization import save_triptych
from src.models.ccunet import CCUNet
from src.utils.io import load_checkpoint, save_json


@torch.no_grad()
def run_inference(
    *,
    ckpt_path: str,
    infer_dir: str,
    latent_path: str,
    modality_names: List[str],
    source_modality: str,
    target_modality: str,
    auto_use_latent_metadata: bool,
    max_samples: int,
    output_dir: str | Path,
    device: torch.device,
    save_visualizations: bool = True,
) -> Dict:
    latent_batch = load_generated_latent_batch(latent_path)
    if auto_use_latent_metadata:
        source_modality = latent_batch.source_modality or source_modality
        target_modality = latent_batch.target_modality or target_modality
    mod_to_id = {m: i for i, m in enumerate(modality_names)}
    if source_modality not in mod_to_id or target_modality not in mod_to_id:
        raise ValueError("Configured source/target modality not found in modality_names.")

    ckpt = load_checkpoint(ckpt_path, map_location=device)
    model = CCUNet(
        latent_dim=int(ckpt.get("latent_dim", latent_batch.latents.shape[1])),
        num_modalities=len(modality_names),
        mod_embed_dim=int(ckpt.get("mod_embed_dim", 16)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    image_transform = build_image_transform(int(ckpt.get("image_size", 128)))
    src_images = []
    for fname in sorted(os.listdir(infer_dir)):
        if not fname.lower().endswith(".png"):
            continue
        parsed = parse_flat_braTS_filename(fname)
        if parsed is None:
            continue
        _, modality, _ = parsed
        if modality == source_modality:
            src_images.append(fname)
    src_images = src_images[:max_samples]
    latents = latent_batch.latents[:max_samples]
    n = min(len(src_images), len(latents))
    src_images = src_images[:n]
    latents = latents[:n]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    src_id_b = torch.tensor([mod_to_id[source_modality]], dtype=torch.long, device=device)
    tgt_id_b = torch.tensor([mod_to_id[target_modality]], dtype=torch.long, device=device)
    for idx, src_fname in tqdm(list(enumerate(src_images)), desc="Inference"):
        src_path = os.path.join(infer_dir, src_fname)
        tgt_fname = src_fname.replace(f"_{source_modality}_", f"_{target_modality}_")
        tgt_path = os.path.join(infer_dir, tgt_fname)
        if not os.path.exists(tgt_path):
            continue
        src = image_transform(Image.open(src_path).convert("L")).unsqueeze(0).to(device)
        tgt = image_transform(Image.open(tgt_path).convert("L")).unsqueeze(0).to(device)
        z = latents[idx].unsqueeze(0).to(device)
        pred = model(src, z, src_id_b, tgt_id_b)
        metric = single_image_metrics(pred[0].cpu(), tgt[0].cpu())
        metric.update({"file": src_fname, "target_file": tgt_fname})
        metrics.append(metric)
        if save_visualizations:
            save_triptych(src[0].cpu(), pred[0].cpu(), tgt[0].cpu(), output_dir / src_fname.replace(f"_{source_modality}_", f"_pred_{target_modality}_"))
    summary = {
        "source_modality": source_modality,
        "target_modality": target_modality,
        "latent_file": latent_path,
        **summarize_metric_list(metrics),
        "metrics": metrics,
    }
    save_json(output_dir / "infer_metrics.json", summary)
    return summary
