from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch



def _to_numpy(x: torch.Tensor):
    if x.dim() == 3:
        x = x.squeeze(0)
    return x.detach().cpu().clamp(0, 1).numpy()



def save_triptych(src: torch.Tensor, pred: torch.Tensor, tgt: torch.Tensor, save_path: str | Path, titles=("Input", "Prediction", "Ground Truth")) -> str:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(_to_numpy(src), cmap="gray")
    axes[1].imshow(_to_numpy(pred), cmap="gray")
    axes[2].imshow(_to_numpy(tgt), cmap="gray")
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    return str(save_path)



def save_history_plots(history: dict, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png")
    plt.close()

    plt.figure(figsize=(7, 5))
    for key in ["train_l1", "val_l1", "train_psnr", "val_psnr", "train_ssim", "val_ssim"]:
        plt.plot(history[key], label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "metric_curves.png")
    plt.close()
