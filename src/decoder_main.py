from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from src.datasets.latent_image_dataset import LatentImageDataset
from src.inference.decode_latents import run_decoder_eval
from src.models.latent_decoder import LatentDecoderV2
from src.training.decoder_trainer import train_decoder
from src.utils.config import load_yaml_config
from src.utils.io import load_checkpoint, save_json
from src.utils.paths import build_experiment_paths
from src.utils.reproducibility import configure_torch, resolve_device, set_seed


def build_loader(ds, cfg, shuffle: bool):
    runtime = cfg["runtime"]
    kwargs = {
        "batch_size": int(cfg["training"]["batch_size"]),
        "shuffle": shuffle,
        "num_workers": int(runtime["num_workers"]),
        "pin_memory": bool(runtime["pin_memory"]),
    }
    if int(runtime["num_workers"]) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(runtime["prefetch_factor"])
    return DataLoader(ds, **kwargs)


def build_model(cfg):
    return LatentDecoderV2(
        latent_dim=int(cfg["model"]["latent_dim"]),
        base_ch=int(cfg["model"]["base_ch"]),
        out_ch=int(cfg["model"]["out_ch"]),
    )


def run_train(cfg):
    set_seed(int(cfg["seed"]))
    device = resolve_device(cfg["device"])
    configure_torch(device)
    exp_paths = build_experiment_paths(cfg["output_root"], cfg["experiment_name"])
    save_json(exp_paths["root"] / "config.json", cfg)
    train_ds = LatentImageDataset(cfg["data"]["train_npz"], img_size=int(cfg["data"]["img_size"]))
    val_ds = LatentImageDataset(cfg["data"]["val_npz"], img_size=int(cfg["data"]["img_size"]))
    cfg["model"]["latent_dim"] = int(train_ds.latents.shape[1])
    train_loader = build_loader(train_ds, cfg, shuffle=True)
    val_loader = build_loader(val_ds, cfg, shuffle=False)
    model = build_model(cfg).to(device)
    history = train_decoder(model, train_loader, val_loader, device, cfg, exp_paths)
    save_json(exp_paths["metrics"] / "final_history.json", history)


def run_eval(cfg):
    set_seed(int(cfg["seed"]))
    device = resolve_device(cfg["device"])
    configure_torch(device)
    exp_paths = build_experiment_paths(cfg["output_root"], cfg["experiment_name"])
    ckpt_name = cfg["evaluation"]["checkpoint"]
    ckpt_path = exp_paths["checkpoints"] / ("best_decoder.pt" if ckpt_name == "best" else "last_decoder.pt")
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    cfg["model"]["latent_dim"] = int(ckpt["latent_dim"])
    cfg["model"]["base_ch"] = int(ckpt.get("base_ch", cfg["model"]["base_ch"]))
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    metrics = run_decoder_eval(model, cfg, device, exp_paths)
    save_json(exp_paths["metrics"] / "eval_summary.json", {"checkpoint": str(ckpt_path), **metrics})


def main():
    parser = argparse.ArgumentParser(description="Decoder entry point")
    parser.add_argument("mode", choices=["train", "eval"])
    parser.add_argument("--config", default="configs/decoder.yaml")
    args = parser.parse_args()
    cfg = load_yaml_config(args.config)
    if args.mode == "train":
        run_train(cfg)
    elif args.mode == "eval":
        run_eval(cfg)


if __name__ == "__main__":
    main()
