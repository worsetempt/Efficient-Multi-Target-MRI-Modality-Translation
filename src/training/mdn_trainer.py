from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from src.datasets.latent_class_dataset import (
    BalancedClassBatchSampler,
    NormalizedLatentDataset,
    load_mrm_latent_artifact,
    split_by_class,
)
from src.evaluation.mdn_metrics import frechet_distance, linear_mmd2
from src.models.mdn_diffusion import EMA, SCMDN, cosine_beta_schedule, extract
from src.utils.io import load_checkpoint, save_checkpoint, save_json


@dataclass
class MDNConfig:
    train_latent_path: str
    eval_latent_path: str
    modality_names: List[str]
    output_root: str
    experiment_name: str
    device: str
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    hidden_dim: int
    max_time_steps: int
    n_blocks: int
    dropout: float
    class_dropout_prob: float
    ema_decay: float
    grad_clip: float
    amp: bool
    resume_ckpt: Optional[str]
    lambda_x0: float
    lambda_proto: float
    lambda_stats: float
    lambda_style_ortho: float
    lambda_style_margin: float
    lambda_class_sep: float
    metrics_num_samples: int


def style_orthogonality_loss(style_weight: torch.Tensor) -> torch.Tensor:
    w = F.normalize(style_weight[:-1], dim=-1)
    gram = w @ w.t()
    return F.mse_loss(gram, torch.eye(gram.shape[0], device=gram.device))


def prototype_margin_loss(prototypes: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    p = F.normalize(prototypes, dim=-1)
    sim = p @ p.t()
    n = sim.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
    return F.relu(sim[mask] - (1.0 - margin)).mean()


def class_stats_loss(pred_x0: torch.Tensor, true_x0: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    losses = []
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() < 2:
            continue
        p = pred_x0[mask]
        y = true_x0[mask]
        mean_loss = F.mse_loss(p.mean(dim=0), y.mean(dim=0))
        p_var = ((p - p.mean(dim=0, keepdim=True)).pow(2).mean(dim=0) + 1e-6).sqrt()
        y_var = ((y - y.mean(dim=0, keepdim=True)).pow(2).mean(dim=0) + 1e-6).sqrt()
        losses.append(mean_loss + F.mse_loss(p_var, y_var))
    return torch.stack(losses).mean() if losses else pred_x0.new_tensor(0.0)


def prototype_pull_loss(pred_x0: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_x0, prototypes[labels])


def cosine_separation_loss(pred_x0: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if pred_x0.shape[0] < 2:
        return pred_x0.new_tensor(0.0)
    x = F.normalize(pred_x0, dim=-1)
    sim = x @ x.t()
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(labels.shape[0], device=labels.device, dtype=torch.bool)
    same = same & ~eye
    diff = (~same) & ~eye
    loss = pred_x0.new_tensor(0.0)
    if same.any():
        loss = loss + (1.0 - sim[same]).mean()
    if diff.any():
        loss = loss + F.relu(sim[diff] - 0.1).mean()
    return loss


class DiffusionTrainer:
    def __init__(self, config: MDNConfig, exp_paths: dict, device: torch.device):
        self.config = config
        self.exp_paths = exp_paths
        self.device = device
        self.num_classes = len(config.modality_names)

        train_view = load_mrm_latent_artifact(config.train_latent_path)
        datasets = split_by_class(train_view, config.modality_names)
        concat_raw = ConcatDataset(datasets)
        all_latents = torch.cat([d.latents for d in datasets], dim=0)
        self.latent_mean = all_latents.mean(dim=0)
        self.latent_std = all_latents.std(dim=0).clamp_min(1e-6)
        self.dataset = NormalizedLatentDataset(concat_raw, self.latent_mean, self.latent_std)

        batch_sampler = BalancedClassBatchSampler(self.dataset.base_dataset, config.batch_size, self.num_classes, drop_last=True)
        self.loader = DataLoader(self.dataset, batch_sampler=batch_sampler, num_workers=0, pin_memory=device.type == "cuda")
        self.latent_dim = datasets[0].latents.shape[1]

        self.model = SCMDN(self.latent_dim, config.hidden_dim, self.num_classes, config.n_blocks, config.dropout, config.class_dropout_prob).to(device)
        self.ema_model = SCMDN(self.latent_dim, config.hidden_dim, self.num_classes, config.n_blocks, config.dropout, 0.0).to(device)
        self.ema_model.load_state_dict(self.model.state_dict())
        self.ema = EMA(self.ema_model, decay=config.ema_decay)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, betas=(0.9, 0.95))
        self.scaler = torch.amp.GradScaler("cuda", enabled=(config.amp and device.type == "cuda"))

        self.betas = cosine_beta_schedule(config.max_time_steps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), self.alphas_cumprod[:-1]], dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.history = {"loss": [], "loss_v": [], "loss_x0": [], "loss_proto": [], "loss_stats": [], "loss_style": [], "loss_sep": [], "fd_mean": []}

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0 + extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise

    def predict_x0_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v

    def predict_eps_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * x_t + extract(self.sqrt_alphas_cumprod, t, x_t.shape) * v

    def ddim_step(self, model, x_t: torch.Tensor, labels: torch.Tensor, t: torch.Tensor, self_cond: Optional[torch.Tensor], cfg_scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
        v_cond = model(x_t, labels, t, self_cond=self_cond, force_drop_label=False)
        if cfg_scale > 1.0:
            v_uncond = model(x_t, labels, t, self_cond=self_cond, force_drop_label=True)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = v_cond
        x0 = self.predict_x0_from_v(x_t, t, v).clamp(-6.0, 6.0)
        eps = self.predict_eps_from_v(x_t, t, v)
        a_prev = extract(self.alphas_cumprod_prev, t, x_t.shape)
        x_prev = torch.sqrt(a_prev) * x0 + torch.sqrt((1.0 - a_prev).clamp_min(1e-8)) * eps
        return x_prev, x0

    def save(self, epoch: int, best_loss: float, is_best: bool) -> None:
        payload = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "ema": self.ema.state_dict(),
            "best_loss": best_loss,
            "config": asdict(self.config),
            "latent_mean": self.latent_mean,
            "latent_std": self.latent_std,
            "history": self.history,
        }
        save_checkpoint(self.exp_paths["checkpoints"] / "last.pt", payload)
        if is_best:
            save_checkpoint(self.exp_paths["checkpoints"] / "best.pt", payload)

    def resume(self) -> float:
        best_loss = float("inf")
        ckpt_path = self.config.resume_ckpt
        if ckpt_path:
            ckpt = load_checkpoint(ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.ema.load_state_dict(ckpt["ema"])
            self.ema.copy_to(self.ema_model)
            self.latent_mean = ckpt.get("latent_mean", self.latent_mean).cpu()
            self.latent_std = ckpt.get("latent_std", self.latent_std).cpu()
            self.history = ckpt.get("history", self.history)
            best_loss = float(ckpt.get("best_loss", float("inf")))
        return best_loss

    @torch.no_grad()
    def generate(self, target_classes: List[int], n_samples_per_class: int, n_sampling_steps: int, cfg_scale: float, ckpt_path: Optional[str] = None, use_ema: bool = True) -> Dict[str, torch.Tensor]:
        if ckpt_path:
            ckpt = load_checkpoint(ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model"])
            if "ema" in ckpt:
                self.ema.load_state_dict(ckpt["ema"])
                self.ema.copy_to(self.ema_model)
            self.latent_mean = ckpt.get("latent_mean", self.latent_mean).cpu()
            self.latent_std = ckpt.get("latent_std", self.latent_std).cpu()
        model = self.ema_model if use_ema else self.model
        if use_ema:
            self.ema.copy_to(self.ema_model)
        model.eval()
        steps = torch.linspace(self.config.max_time_steps - 1, 0, n_sampling_steps, device=self.device).long()
        outputs: Dict[str, torch.Tensor] = {}
        mean = self.latent_mean.to(self.device)
        std = self.latent_std.to(self.device)
        for cls in target_classes:
            labels = torch.full((n_samples_per_class,), cls, device=self.device, dtype=torch.long)
            x_t = torch.randn(n_samples_per_class, self.latent_dim, device=self.device)
            self_cond = None
            for step in tqdm(steps, desc=f"Sampling class {cls}", leave=False):
                t = torch.full((n_samples_per_class,), int(step.item()), device=self.device, dtype=torch.long)
                x_t, pred_x0 = self.ddim_step(model, x_t, labels, t, self_cond, cfg_scale)
                self_cond = pred_x0
            outputs[str(cls)] = (x_t * std + mean).detach().cpu()
        return outputs

    @torch.no_grad()
    def quick_fd(self, samples_per_class: int) -> Dict[str, float]:
        eval_view = load_mrm_latent_artifact(self.config.eval_latent_path)
        out: Dict[str, float] = {}
        fd_values = []
        generated = self.generate(list(range(self.num_classes)), min(samples_per_class, self.config.metrics_num_samples), min(32, self.config.max_time_steps), 2.0, use_ema=True)
        for cls in range(self.num_classes):
            real = eval_view.latents[eval_view.labels == cls][:samples_per_class].numpy()
            gen = generated[str(cls)][: len(real)].numpy()
            fd = frechet_distance(real, gen)
            out[f"fd_cls_{cls}"] = fd
            out[f"mmd_cls_{cls}"] = linear_mmd2(real, gen)
            fd_values.append(fd)
        out["fd_mean"] = float(np.mean(fd_values)) if fd_values else float("nan")
        return out

    def train(self) -> Dict[str, list]:
        best_loss = self.resume()
        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            stats = {k: 0.0 for k in ["loss", "loss_v", "loss_x0", "loss_proto", "loss_stats", "loss_style", "loss_sep"]}
            pbar = tqdm(self.loader, desc=f"Epoch {epoch:03d}/{self.config.epochs}")
            for x0, labels in pbar:
                x0 = x0.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                bsz = x0.shape[0]
                t = torch.randint(0, self.config.max_time_steps, (bsz,), device=self.device).long()
                noise = torch.randn_like(x0)
                x_t = self.q_sample(x0, t, noise)
                v_target = extract(self.sqrt_alphas_cumprod, t, x0.shape) * noise - extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * x0
                with torch.no_grad():
                    self_cond = None if np.random.rand() < 0.5 else self.predict_x0_from_v(x_t, t, self.model(x_t, labels, t, self_cond=None)).detach()
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, enabled=(self.config.amp and self.device.type == "cuda")):
                    v_pred = self.model(x_t, labels, t, self_cond=self_cond)
                    pred_x0 = self.predict_x0_from_v(x_t, t, v_pred)
                    loss_v = F.mse_loss(v_pred, v_target)
                    loss_x0 = F.mse_loss(pred_x0, x0)
                    loss_proto = prototype_pull_loss(pred_x0, labels, self.model.prototype_table)
                    loss_stats = class_stats_loss(pred_x0, x0, labels, self.num_classes)
                    loss_style = style_orthogonality_loss(self.model.class_embed.weight)
                    loss_margin = prototype_margin_loss(self.model.prototype_table)
                    loss_sep = cosine_separation_loss(pred_x0, labels)
                    total_loss = loss_v + self.config.lambda_x0 * loss_x0 + self.config.lambda_proto * loss_proto + self.config.lambda_stats * loss_stats + self.config.lambda_style_ortho * loss_style + self.config.lambda_style_margin * loss_margin + self.config.lambda_class_sep * loss_sep
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.ema.update(self.model)
                stats["loss"] += total_loss.item()
                stats["loss_v"] += loss_v.item()
                stats["loss_x0"] += loss_x0.item()
                stats["loss_proto"] += loss_proto.item()
                stats["loss_stats"] += loss_stats.item()
                stats["loss_style"] += (loss_style.item() + loss_margin.item())
                stats["loss_sep"] += loss_sep.item()
                pbar.set_postfix(loss=f"{total_loss.item():.4f}")
            n_batches = max(1, len(self.loader))
            for k in stats:
                stats[k] /= n_batches
                self.history[k].append(stats[k])
            quick = self.quick_fd(min(128, self.config.metrics_num_samples))
            self.history["fd_mean"].append(quick["fd_mean"])
            improved = stats["loss"] < best_loss
            if improved:
                best_loss = stats["loss"]
            self.save(epoch, best_loss, improved)
            save_json(self.exp_paths["metrics"] / "train_metrics.json", {"history": self.history, "latest": {**stats, **quick, "epoch": epoch, "best_loss": best_loss}})
        return self.history
