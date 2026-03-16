from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.latent_image_dataset import GeneratedLatentEvalDataset
from src.evaluation.decoder_metrics import evaluate_decoder
from src.evaluation.decoder_visualization import save_decoded_outputs
from src.utils.config import dump_json


def run_decoder_eval(model, cfg: dict, device: torch.device, exp_paths: dict):
    eval_cfg = cfg["evaluation"]
    ds = GeneratedLatentEvalDataset(
        latent_path=eval_cfg["latent_path"],
        reference_npz=eval_cfg.get("reference_npz"),
        img_size=int(cfg["data"]["img_size"]),
    )
    loader = DataLoader(
        ds,
        batch_size=int(eval_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["runtime"]["num_workers"]),
        pin_memory=bool(cfg["runtime"]["pin_memory"]),
    )
    visuals_dir = Path(exp_paths["visuals"]) / "decoded"
    save_decoded_outputs(model, loader, device, visuals_dir, amp=bool(cfg["runtime"]["amp"]), max_items=eval_cfg.get("max_visualizations"))

    has_targets = any(target is not None for _, target, _, _ in [ds[i] for i in range(min(len(ds), 4))]) if len(ds) > 0 else False
    metrics = {"decoded_dir": str(visuals_dir), "num_samples": len(ds)}
    if has_targets:
        metrics.update(evaluate_decoder(model, loader, device, amp=bool(cfg["runtime"]["amp"])))
    dump_json(exp_paths["metrics"] / "eval_generated.json", metrics)
    return metrics
