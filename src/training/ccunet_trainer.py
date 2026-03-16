from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.ccunet_dataset import AllPairsMRMDataset, make_sample_subsets
from src.evaluation.ccunet_metrics import batch_l1, batch_psnr, batch_ssim
from src.evaluation.ccunet_visualization import save_history_plots, save_triptych
from src.models.ccunet import CCUNet
from src.utils.io import load_checkpoint, save_checkpoint, save_json


@dataclass
class CCUNetConfig:
    train_image_dir: str
    train_mrm_latent_path: str
    modality_names: list[str]
    output_root: str
    experiment_name: str
    device: str = "cuda"
    seed: int = 42
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_workers: int = 4
    persistent_workers: bool = True
    image_size: int = 128
    latent_dim: int = 768
    mod_embed_dim: int = 16
    train_samples_per_class: int = 3000
    val_total_samples: int = 2000
    split_seed: int = 42


class CCUNetTrainer:
    def __init__(self, cfg: CCUNetConfig, exp_paths: dict, device: torch.device):
        self.cfg = cfg
        self.exp_paths = exp_paths
        self.device = device
        full_dataset = AllPairsMRMDataset(
            root_dir=cfg.train_image_dir,
            mrm_latent_path=cfg.train_mrm_latent_path,
            modalities=cfg.modality_names,
            image_size=cfg.image_size,
        )
        train_dataset, val_dataset = make_sample_subsets(
            full_dataset,
            train_per_target=cfg.train_samples_per_class,
            val_total=cfg.val_total_samples,
            seed=cfg.split_seed,
        )
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=cfg.persistent_workers if cfg.num_workers > 0 else False,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=cfg.persistent_workers if cfg.num_workers > 0 else False,
        )
        self.model = CCUNet(latent_dim=cfg.latent_dim, num_modalities=len(cfg.modality_names), mod_embed_dim=cfg.mod_embed_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        self.loss_fn = nn.L1Loss()
        self.history = {k: [] for k in ["train_loss", "val_loss", "train_l1", "train_psnr", "train_ssim", "val_l1", "val_psnr", "val_ssim"]}

    def _step_batch(self, batch: Dict[str, torch.Tensor], train: bool) -> Dict[str, float]:
        src = batch["src"].to(self.device, non_blocking=True)
        tgt = batch["tgt"].to(self.device, non_blocking=True)
        z = batch["tgt_latent"].to(self.device, non_blocking=True)
        src_mod_id = batch["src_mod_id"].to(self.device, non_blocking=True)
        tgt_mod_id = batch["tgt_mod_id"].to(self.device, non_blocking=True)
        if train:
            self.optimizer.zero_grad(set_to_none=True)
        pred = self.model(src, z, src_mod_id, tgt_mod_id)
        loss = self.loss_fn(pred, tgt)
        if train:
            loss.backward()
            self.optimizer.step()
        return {
            "loss": float(loss.item()),
            "l1": batch_l1(pred, tgt),
            "psnr": batch_psnr(pred, tgt),
            "ssim": batch_ssim(pred, tgt),
        }

    def _run_epoch(self, loader: DataLoader, train: bool) -> Dict[str, float]:
        self.model.train(mode=train)
        totals = {"loss": 0.0, "l1": 0.0, "psnr": 0.0, "ssim": 0.0}
        iterator = tqdm(loader, desc="train" if train else "val", leave=False)
        for batch in iterator:
            with torch.set_grad_enabled(train):
                metrics = self._step_batch(batch, train=train)
            for k in totals:
                totals[k] += metrics[k]
            iterator.set_postfix(loss=f"{metrics['loss']:.4f}", l1=f"{metrics['l1']:.4f}", psnr=f"{metrics['psnr']:.2f}", ssim=f"{metrics['ssim']:.4f}")
        n = max(len(loader), 1)
        return {k: v / n for k, v in totals.items()}

    def train(self) -> None:
        best_val = float("inf")
        for epoch in range(self.cfg.epochs):
            train_metrics = self._run_epoch(self.train_loader, train=True)
            val_metrics = self._run_epoch(self.val_loader, train=False)
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_l1"].append(train_metrics["l1"])
            self.history["train_psnr"].append(train_metrics["psnr"])
            self.history["train_ssim"].append(train_metrics["ssim"])
            self.history["val_l1"].append(val_metrics["l1"])
            self.history["val_psnr"].append(val_metrics["psnr"])
            self.history["val_ssim"].append(val_metrics["ssim"])
            save_json(self.exp_paths["metrics"] / "history.json", self.history)
            save_history_plots(self.history, self.exp_paths["visuals"])
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                save_checkpoint(self.exp_paths["checkpoints"] / "best.pt", {"model": self.model.state_dict(), "latent_dim": self.cfg.latent_dim, "modality_names": self.cfg.modality_names, "image_size": self.cfg.image_size, "mod_embed_dim": self.cfg.mod_embed_dim})
            save_checkpoint(self.exp_paths["checkpoints"] / "last.pt", {"model": self.model.state_dict(), "latent_dim": self.cfg.latent_dim, "modality_names": self.cfg.modality_names, "image_size": self.cfg.image_size, "mod_embed_dim": self.cfg.mod_embed_dim})

            sample = self.val_dataset[0]
            with torch.no_grad():
                pred = self.model(
                    sample["src"].unsqueeze(0).to(self.device),
                    sample["tgt_latent"].unsqueeze(0).to(self.device),
                    sample["src_mod_id"].unsqueeze(0).to(self.device),
                    sample["tgt_mod_id"].unsqueeze(0).to(self.device),
                )[0].cpu()
            save_triptych(sample["src"], pred, sample["tgt"], self.exp_paths["visuals"] / f"epoch_{epoch+1:03d}.png")

    def load(self, checkpoint_path: str | Path) -> None:
        ckpt = load_checkpoint(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
