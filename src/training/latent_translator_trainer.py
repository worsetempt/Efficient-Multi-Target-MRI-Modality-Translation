from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.aligned_latent_pairs import AllPairsLatentDataset, MultiModalityLatentStore
from src.models.latent_translator import EMA, LatentTranslator
from src.utils.io import load_checkpoint, save_checkpoint, save_json


@dataclass
class TranslatorConfig:
    train_latent_paths: Dict[str, str]
    val_latent_paths: Dict[str, str]
    test_latent_paths: Optional[Dict[str, str]]
    modality_names: List[str]
    output_root: str
    experiment_name: str
    device: str
    seed: int
    epochs: int
    batch_size: int
    num_workers: int
    learning_rate: float
    weight_decay: float
    grad_clip: float
    amp: bool
    resume_ckpt: Optional[str]
    save_every: int
    ema_decay: float
    hidden_dim: int
    n_blocks: int
    dropout: float
    use_layernorm: bool
    lambda_l1: float
    lambda_cos: float
    lambda_delta: float
    lambda_pair_consistency: float
    lambda_centroid: float
    max_translate_samples: Optional[int]


def cosine_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(x, y, dim=-1)).mean()


def pair_consistency_loss(z_pred: torch.Tensor, z_src: torch.Tensor, z_tgt: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(z_pred - z_src, z_tgt - z_src)


def centroid_match_loss(z_pred: torch.Tensor, z_tgt: torch.Tensor, tgt_id: torch.Tensor, num_modalities: int) -> torch.Tensor:
    loss = z_pred.new_tensor(0.0)
    count = 0
    for m in range(num_modalities):
        mask = tgt_id == m
        if mask.sum() >= 2:
            loss = loss + F.mse_loss(z_pred[mask].mean(dim=0), z_tgt[mask].mean(dim=0))
            count += 1
    return loss / count if count else z_pred.new_tensor(0.0)


def latent_psnr_from_mse(mse: float, data_range: float = 2.0) -> float:
    return 10.0 * torch.log10(torch.tensor((data_range ** 2) / max(float(mse), 1e-12))).item()


class TranslatorTrainer:
    def __init__(self, config: TranslatorConfig, exp_paths: dict, device: torch.device):
        self.config = config
        self.exp_paths = exp_paths
        self.device = device
        self.modality_names = config.modality_names
        self.modality_to_id = {m: i for i, m in enumerate(self.modality_names)}
        self.num_modalities = len(self.modality_names)

        self.train_store = MultiModalityLatentStore.from_paths(config.train_latent_paths, self.modality_names)
        self.val_store = MultiModalityLatentStore.from_paths(config.val_latent_paths, self.modality_names)
        self.test_store = MultiModalityLatentStore.from_paths(config.test_latent_paths, self.modality_names) if config.test_latent_paths else None
        self.latent_dim = self.train_store.latent_dim

        all_train = torch.cat([self.train_store.get(m) for m in self.modality_names], dim=0)
        self.latent_mean = all_train.mean(dim=0)
        self.latent_std = all_train.std(dim=0).clamp_min(1e-6)

        self.train_ds = AllPairsLatentDataset(self.train_store, self.modality_to_id, self.latent_mean, self.latent_std)
        self.val_ds = AllPairsLatentDataset(self.val_store, self.modality_to_id, self.latent_mean, self.latent_std)
        self.train_loader = DataLoader(self.train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=device.type == "cuda")
        self.val_loader = DataLoader(self.val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=device.type == "cuda")

        self.model = LatentTranslator(self.latent_dim, self.num_modalities, config.hidden_dim, config.n_blocks, config.dropout, config.use_layernorm).to(device)
        self.ema_model = LatentTranslator(self.latent_dim, self.num_modalities, config.hidden_dim, config.n_blocks, config.dropout, config.use_layernorm).to(device)
        self.ema_model.load_state_dict(self.model.state_dict())
        self.ema = EMA(self.ema_model, config.ema_decay)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scaler = torch.amp.GradScaler("cuda", enabled=(config.amp and device.type == "cuda"))
        self.history = {"train_loss": [], "val_loss": [], "val_mse": [], "val_cos": [], "val_psnr_proxy": []}

    def _compute_loss(self, batch, model):
        z_src = batch["z_src"].to(self.device, non_blocking=True)
        z_tgt = batch["z_tgt"].to(self.device, non_blocking=True)
        src_id = batch["src_id"].to(self.device, non_blocking=True)
        tgt_id = batch["tgt_id"].to(self.device, non_blocking=True)
        z_pred, delta_pred = model(z_src, src_id, tgt_id)
        delta_true = z_tgt - z_src
        loss_mse = F.mse_loss(z_pred, z_tgt)
        loss_l1 = F.smooth_l1_loss(z_pred, z_tgt)
        loss_cos = cosine_loss(z_pred, z_tgt)
        loss_delta = F.smooth_l1_loss(delta_pred, delta_true)
        loss_pair = pair_consistency_loss(z_pred, z_src, z_tgt)
        loss_centroid = centroid_match_loss(z_pred, z_tgt, tgt_id, self.num_modalities)
        total = loss_mse + self.config.lambda_l1 * loss_l1 + self.config.lambda_cos * loss_cos + self.config.lambda_delta * loss_delta + self.config.lambda_pair_consistency * loss_pair + self.config.lambda_centroid * loss_centroid
        metrics = {
            "loss": total.detach(),
            "loss_mse": loss_mse.detach(),
            "loss_l1": loss_l1.detach(),
            "loss_cos": loss_cos.detach(),
            "loss_delta": loss_delta.detach(),
            "loss_pair": loss_pair.detach(),
            "loss_centroid": loss_centroid.detach(),
        }
        return total, metrics

    def save(self, epoch: int, best_val_loss: float, is_best: bool) -> None:
        payload = {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "history": self.history,
            "latent_mean": self.latent_mean.cpu(),
            "latent_std": self.latent_std.cpu(),
            "config": asdict(self.config),
            "modality_to_id": self.modality_to_id,
        }
        save_checkpoint(self.exp_paths["checkpoints"] / "last.pt", payload)
        if is_best:
            save_checkpoint(self.exp_paths["checkpoints"] / "best.pt", payload)
        if epoch % self.config.save_every == 0:
            save_checkpoint(self.exp_paths["checkpoints"] / f"checkpoint_epoch_{epoch:03d}.pt", payload)

    def resume(self) -> Tuple[int, float]:
        if not self.config.resume_ckpt:
            return 0, float("inf")
        ckpt = load_checkpoint(self.config.resume_ckpt, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if "ema" in ckpt:
            self.ema.load_state_dict(ckpt["ema"])
            self.ema.copy_to(self.ema_model)
        if "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])
        self.history = ckpt.get("history", self.history)
        self.latent_mean = ckpt.get("latent_mean", self.latent_mean).cpu()
        self.latent_std = ckpt.get("latent_std", self.latent_std).cpu()
        return int(ckpt.get("epoch", 0)), float(ckpt.get("best_val_loss", float("inf")))

    def evaluate(self, use_ema: bool = True) -> Dict[str, float]:
        if use_ema:
            self.ema.copy_to(self.ema_model)
            model = self.ema_model
        else:
            model = self.model
        model.eval()
        sums = {"loss": 0.0, "loss_mse": 0.0, "loss_l1": 0.0, "loss_cos": 0.0, "loss_delta": 0.0, "loss_pair": 0.0, "loss_centroid": 0.0}
        n_batches = 0
        with torch.no_grad():
            for batch in self.val_loader:
                with torch.autocast(device_type=self.device.type, enabled=(self.config.amp and self.device.type == "cuda")):
                    _, metrics = self._compute_loss(batch, model)
                for k in sums:
                    sums[k] += float(metrics[k].item())
                n_batches += 1
        for k in sums:
            sums[k] /= max(n_batches, 1)
        sums["psnr_proxy"] = latent_psnr_from_mse(sums["loss_mse"])
        return sums

    def train(self) -> None:
        start_epoch, best_val_loss = self.resume()
        for epoch in range(start_epoch + 1, self.config.epochs + 1):
            self.model.train()
            epoch_sums = {"loss": 0.0, "loss_mse": 0.0, "loss_l1": 0.0, "loss_cos": 0.0, "loss_delta": 0.0, "loss_pair": 0.0, "loss_centroid": 0.0}
            n_batches = 0
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:03d}/{self.config.epochs}")
            for batch in pbar:
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, enabled=(self.config.amp and self.device.type == "cuda")):
                    total_loss, metrics = self._compute_loss(batch, self.model)
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.ema.update(self.model)
                for k in epoch_sums:
                    epoch_sums[k] += float(metrics[k].item())
                n_batches += 1
                pbar.set_postfix(loss=f"{metrics['loss'].item():.4f}", mse=f"{metrics['loss_mse'].item():.4f}", cos=f"{metrics['loss_cos'].item():.4f}")
            for k in epoch_sums:
                epoch_sums[k] /= max(n_batches, 1)
            val_metrics = self.evaluate(use_ema=True)
            self.history["train_loss"].append(epoch_sums["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_mse"].append(val_metrics["loss_mse"])
            self.history["val_cos"].append(val_metrics["loss_cos"])
            self.history["val_psnr_proxy"].append(val_metrics["psnr_proxy"])
            improved = val_metrics["loss"] < best_val_loss
            if improved:
                best_val_loss = val_metrics["loss"]
            self.save(epoch, best_val_loss, improved)
            save_json(self.exp_paths["metrics"] / "history.json", {"history": self.history, "best_val_loss": best_val_loss})

    def load_for_inference(self, ckpt_path: Optional[str], use_ema: bool = True):
        if ckpt_path:
            ckpt = load_checkpoint(ckpt_path, map_location=self.device)
        else:
            ckpt = load_checkpoint(self.exp_paths["checkpoints"] / "best.pt", map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        if "ema" in ckpt:
            self.ema.load_state_dict(ckpt["ema"])
            self.ema.copy_to(self.ema_model)
        self.latent_mean = ckpt.get("latent_mean", self.latent_mean).cpu()
        self.latent_std = ckpt.get("latent_std", self.latent_std).cpu()
        model = self.ema_model if use_ema else self.model
        model.eval()
        return model

    def get_store_for_split(self, split: str) -> MultiModalityLatentStore:
        split = split.lower()
        if split == "train":
            return self.train_store
        if split == "val":
            return self.val_store
        if split == "test" and self.test_store is not None:
            return self.test_store
        raise ValueError(f"Unsupported split '{split}'")
