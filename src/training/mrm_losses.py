from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_smooth_l1(recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    region = mask > 0.5
    if region.sum() < 1:
        return torch.tensor(0.0, device=recon.device)
    return F.smooth_l1_loss(recon[region], target[region], reduction="mean")


def full_smooth_l1(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(recon, target, reduction="mean")


def grad_loss_l1(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dx_r = recon[..., :, 1:] - recon[..., :, :-1]
    dx_t = target[..., :, 1:] - target[..., :, :-1]
    dy_r = recon[..., 1:, :] - recon[..., :-1, :]
    dy_t = target[..., 1:, :] - target[..., :-1, :]
    return (dx_r - dx_t).abs().mean() + (dy_r - dy_t).abs().mean()


def loss_mrm_only(recon, target, mask, alpha=0.1, beta=0.2):
    l_masked = masked_smooth_l1(recon, target, mask)
    l_full = full_smooth_l1(recon, target)
    l_grad = grad_loss_l1(recon, target)
    return l_masked + alpha * l_full + beta * l_grad


def loss_mrm_plus_cls(recon, target, mask, z, labels, aux_head, lambda_cls, alpha=0.1, beta=0.2):
    l_mrm = loss_mrm_only(recon, target, mask, alpha=alpha, beta=beta)
    logits = aux_head(z)
    l_cls = F.cross_entropy(logits, labels)
    return l_mrm + lambda_cls * l_cls
