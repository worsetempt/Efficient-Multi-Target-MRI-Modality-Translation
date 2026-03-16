from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from src.training.latent_translator_trainer import TranslatorTrainer
from src.utils.latent_contracts import validate_latent_payload


@torch.no_grad()
def generate_translated_latents(trainer: TranslatorTrainer, split: str, ckpt_path: Optional[str], use_ema: bool) -> Dict[str, torch.Tensor]:
    model = trainer.load_for_inference(ckpt_path, use_ema=use_ema)
    store = trainer.get_store_for_split(split)
    mean = trainer.latent_mean.to(trainer.device)
    std = trainer.latent_std.to(trainer.device)
    max_n = len(store) if trainer.config.max_translate_samples is None else min(len(store), int(trainer.config.max_translate_samples))
    outputs: Dict[str, torch.Tensor] = {}
    for src_m in trainer.modality_names:
        for tgt_m in trainer.modality_names:
            if src_m == tgt_m:
                continue
            src_id = trainer.modality_to_id[src_m]
            tgt_id = trainer.modality_to_id[tgt_m]
            z_src_raw = store.get(src_m)[:max_n]
            z_src = ((z_src_raw - trainer.latent_mean) / trainer.latent_std).to(trainer.device)
            src_ids = torch.full((max_n,), src_id, device=trainer.device, dtype=torch.long)
            tgt_ids = torch.full((max_n,), tgt_id, device=trainer.device, dtype=torch.long)
            preds = []
            bs = max(1, trainer.config.batch_size)
            for s in range(0, max_n, bs):
                e = min(s + bs, max_n)
                z_pred_norm, _ = model(z_src[s:e], src_ids[s:e], tgt_ids[s:e])
                preds.append((z_pred_norm * std + mean).detach().cpu())
            outputs[f"{src_m}_to_{tgt_m}"] = torch.cat(preds, dim=0)
    return outputs


def save_translated_outputs(outputs: Dict[str, torch.Tensor], trainer: TranslatorTrainer, split: str, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    store = trainer.get_store_for_split(split)
    summary = {"split": split, "pairs": []}
    max_n = len(next(iter(outputs.values()))) if outputs else 0
    for pair_name, latents in outputs.items():
        src_m, tgt_m = pair_name.split("_to_")
        tgt_raw = store.get(tgt_m)[: len(latents)]
        payload = {
            "latents": latents.float(),
            "source_modality": src_m,
            "target_modality": tgt_m,
            "source_modality_id": trainer.modality_to_id[src_m],
            "target_modality_id": trainer.modality_to_id[tgt_m],
            "split": split,
            "num_samples": int(latents.shape[0]),
            "sample_ids": store.get_sample_ids(src_m)[: len(latents)] if store.get_sample_ids(src_m) is not None else None,
            "source": "latent_translator",
            "metrics": {
                "latent_mse": float(F.mse_loss(latents, tgt_raw).item()),
                "latent_cosine": float(F.cosine_similarity(latents, tgt_raw, dim=-1).mean().item()),
            },
        }
        validate_latent_payload(payload)
        out_path = output_dir / f"{pair_name}.pt"
        torch.save(payload, out_path)
        summary["pairs"].append({"pair": pair_name, "file": str(out_path), **payload["metrics"]})
    torch.save(summary, output_dir / "translation_summary.pt")
    return output_dir
