from __future__ import annotations

from copy import deepcopy

import torch
from tqdm import tqdm

from src.evaluation.mrm_latent_eval import maybe_linear_probe, maybe_silhouette
from src.inference.mrm_extract import extract_latents
from src.training.mrm_losses import loss_mrm_only, loss_mrm_plus_cls
from src.utils.io import save_checkpoint, save_json


def train_epoch(model, loader, optimizer, device, cfg, scaler=None, epoch: int = 0):
    model.train()
    total = 0.0
    n = 0
    it = tqdm(loader, desc=f"Train E{epoch}", leave=False)
    for batch in it:
        masked, target, mask, mod_ids, _ = batch
        masked = masked.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        mod_ids = mod_ids.to(device=device, dtype=torch.long, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        use_amp = scaler is not None and device.type == "cuda"
        if use_amp:
            with torch.amp.autocast("cuda"):
                recon, z = model(masked, return_latent=True)
                if cfg["model"]["loss_type"] == "mrm_only":
                    loss = loss_mrm_only(recon, target, mask, alpha=cfg["model"]["alpha_full"], beta=cfg["model"]["beta_grad"])
                else:
                    loss = loss_mrm_plus_cls(
                        recon, target, mask, z, mod_ids, model.aux_head,
                        lambda_cls=cfg["model"]["lambda_cls"],
                        alpha=cfg["model"]["alpha_full"], beta=cfg["model"]["beta_grad"]
                    )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            recon, z = model(masked, return_latent=True)
            if cfg["model"]["loss_type"] == "mrm_only":
                loss = loss_mrm_only(recon, target, mask, alpha=cfg["model"]["alpha_full"], beta=cfg["model"]["beta_grad"])
            else:
                loss = loss_mrm_plus_cls(
                    recon, target, mask, z, mod_ids, model.aux_head,
                    lambda_cls=cfg["model"]["lambda_cls"],
                    alpha=cfg["model"]["alpha_full"], beta=cfg["model"]["beta_grad"]
                )
            loss.backward()
            optimizer.step()

        total += float(loss.item())
        n += 1
        it.set_postfix(loss=total / max(n, 1))
    return total / max(n, 1)


@torch.no_grad()
def validate_epoch(model, loader, device, cfg, epoch: int = 0):
    model.eval()
    total = 0.0
    n = 0
    it = tqdm(loader, desc=f"Val E{epoch}", leave=False)
    for batch in it:
        masked, target, mask, mod_ids, _ = batch
        masked = masked.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        mod_ids = mod_ids.to(device=device, dtype=torch.long, non_blocking=True)
        if cfg["model"]["loss_type"] == "mrm_only":
            recon = model(masked)
            loss = loss_mrm_only(recon, target, mask, alpha=cfg["model"]["alpha_full"], beta=cfg["model"]["beta_grad"])
        else:
            recon, z = model(masked, return_latent=True)
            loss = loss_mrm_plus_cls(
                recon, target, mask, z, mod_ids, model.aux_head,
                lambda_cls=cfg["model"]["lambda_cls"],
                alpha=cfg["model"]["alpha_full"], beta=cfg["model"]["beta_grad"]
            )
        total += float(loss.item())
        n += 1
        it.set_postfix(loss=total / max(n, 1))
    return total / max(n, 1)


def run_training(model, train_loader, val_loader, device, cfg, exp_paths: dict):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda") if (cfg["runtime"]["amp"] and device.type == "cuda") else None

    history = {"train_loss": [], "val_loss": [], "silhouette": [], "linear_probe": []}
    best_val = float("inf")
    best_state = None

    epochs = int(cfg["training"]["epochs"])
    for ep in range(1, epochs + 1):
        tr = train_epoch(model, train_loader, optimizer, device, cfg, scaler=scaler, epoch=ep)
        va = validate_epoch(model, val_loader, device, cfg, epoch=ep)
        history["train_loss"].append(tr)
        history["val_loss"].append(va)

        if (ep % int(cfg["training"]["eval_every"]) == 0) or ep == epochs:
            lat, lab, _ = extract_latents(model, val_loader, device, max_samples=cfg["evaluation"]["umap"]["max_samples"])
            if cfg["evaluation"]["silhouette"]["enabled"]:
                sil = maybe_silhouette(lat, lab)
                if sil is not None:
                    history["silhouette"].append((ep, sil))
            if cfg["evaluation"]["linear_probe"]["enabled"]:
                probe = maybe_linear_probe(lat, lab, cv=int(cfg["evaluation"]["linear_probe"]["cv"]))
                if probe is not None:
                    history["linear_probe"].append((ep, probe["mean"], probe["std"]))

        ckpt_payload = {
            "epoch": ep,
            "model_state": model.state_dict(),
            "config": cfg,
            "history": history,
        }
        save_checkpoint(exp_paths["checkpoints"] / "last.pt", ckpt_payload)
        if va < best_val:
            best_val = va
            best_state = deepcopy(model.state_dict())
            save_checkpoint(exp_paths["checkpoints"] / "best.pt", ckpt_payload)

        save_json(exp_paths["metrics"] / "train_history.json", history)

    if best_state is not None:
        model.load_state_dict(best_state)
    return history
