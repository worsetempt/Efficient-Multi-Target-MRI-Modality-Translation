from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from src.datasets.brats_mrm_dataset import BraTSMAEDataset, MODALITIES
from src.evaluation.mrm_latent_eval import maybe_linear_probe, maybe_silhouette
from src.evaluation.mrm_metrics import compute_reconstruction_quality
from src.evaluation.mrm_visualization import plot_umap, save_reconstruction_examples
from src.inference.mrm_extract import extract_all_splits, extract_latents
from src.models.swin_mae import SwinMAE
from src.training.mrm_trainer import run_training
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


def build_datasets(cfg):
    data_cfg = cfg["data"]
    root = Path(data_cfg["root_dir"])
    max_pm = data_cfg["max_per_modality"]
    common = {
        "modalities": cfg["modality_names"],
        "mask_ratio": data_cfg["mask_ratio"],
        "patch_size": data_cfg["patch_size"],
        "img_size": data_cfg["img_size"],
    }
    train_ds = BraTSMAEDataset(root / data_cfg["train_dir"], max_per_modality=int(max_pm["train"]), seed=cfg["seed"], **common)
    val_ds = BraTSMAEDataset(root / data_cfg["val_dir"], max_per_modality=int(max_pm["val"]), seed=cfg["seed"] + 1, **common)
    test_ds = BraTSMAEDataset(root / data_cfg["test_dir"], max_per_modality=int(max_pm["test"]), seed=cfg["seed"] + 2, **common)
    return {"train": train_ds, "val": val_ds, "test": test_ds}


def build_model(cfg):
    loss_type = cfg["model"]["loss_type"]
    num_classes = len(cfg["modality_names"]) if loss_type == "mrm_plus_cls" else 0
    return SwinMAE(
        architecture=cfg["model"]["architecture"],
        in_chans=int(cfg["model"]["in_chans"]),
        pretrained=bool(cfg["model"]["pretrained"]),
        num_classes=num_classes,
    )


def load_model_weights(model, exp_paths, checkpoint_name: str, device):
    ckpt_path = exp_paths["checkpoints"] / ("best.pt" if checkpoint_name == "best" else "last.pt")
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return ckpt_path, ckpt


def run_train(cfg):
    set_seed(int(cfg["seed"]))
    device = resolve_device(cfg["device"])
    configure_torch(device)
    exp_paths = build_experiment_paths(cfg["output_root"], cfg["experiment_name"])
    save_json(exp_paths["root"] / "config.json", cfg)

    datasets = build_datasets(cfg)
    train_loader = build_loader(datasets["train"], cfg, shuffle=True)
    val_loader = build_loader(datasets["val"], cfg, shuffle=False)

    model = build_model(cfg).to(device)
    history = run_training(model, train_loader, val_loader, device, cfg, exp_paths)
    save_json(exp_paths["metrics"] / "final_history.json", history)


def run_eval(cfg):
    set_seed(int(cfg["seed"]))
    device = resolve_device(cfg["device"])
    configure_torch(device)
    exp_paths = build_experiment_paths(cfg["output_root"], cfg["experiment_name"])

    datasets = build_datasets(cfg)
    split = cfg["evaluation"]["split"]
    loader = build_loader(datasets[split], cfg, shuffle=False)

    model = build_model(cfg).to(device)
    ckpt_path, ckpt = load_model_weights(model, exp_paths, cfg["evaluation"]["checkpoint"], device)

    recon = compute_reconstruction_quality(model, loader, device, max_batches=int(cfg["evaluation"]["reconstruction"]["max_batches"]))
    lat, lab, _ = extract_latents(model, loader, device, max_samples=int(cfg["evaluation"]["umap"]["max_samples"]))

    payload = {
        "checkpoint": str(ckpt_path),
        "split": split,
        "reconstruction_quality": recon,
    }
    if cfg["evaluation"]["umap"]["enabled"]:
        umap_path = plot_umap(
            lat, lab, cfg["modality_names"],
            exp_paths["visuals"] / f"umap_{split}.png",
            title=f"MRM UMAP ({split})",
            n_neighbors=int(cfg["evaluation"]["umap"]["n_neighbors"]),
            min_dist=float(cfg["evaluation"]["umap"]["min_dist"]),
        )
        payload["umap_path"] = umap_path
    if cfg["evaluation"]["silhouette"]["enabled"]:
        sil = maybe_silhouette(lat, lab)
        if sil is not None:
            payload["silhouette_score"] = sil
    if cfg["evaluation"]["linear_probe"]["enabled"]:
        probe = maybe_linear_probe(lat, lab, cv=int(cfg["evaluation"]["linear_probe"]["cv"]))
        if probe is not None:
            payload["linear_probe"] = probe
    if cfg["evaluation"]["reconstruction"]["save_examples"]:
        save_reconstruction_examples(
            model, loader, device,
            exp_paths["visuals"] / f"recon_examples_{split}",
            n=int(cfg["evaluation"]["reconstruction"]["num_examples"]),
        )
    save_json(exp_paths["metrics"] / f"eval_{split}.json", payload)


def run_extract(cfg):
    set_seed(int(cfg["seed"]))
    device = resolve_device(cfg["device"])
    configure_torch(device)
    exp_paths = build_experiment_paths(cfg["output_root"], cfg["experiment_name"])

    datasets = build_datasets(cfg)
    loaders = {split: build_loader(ds, cfg, shuffle=False) for split, ds in datasets.items()}

    model = build_model(cfg).to(device)
    load_model_weights(model, exp_paths, cfg["evaluation"]["checkpoint"], device)
    summary = extract_all_splits(model, loaders, device, exp_paths["latents"], cfg["modality_names"])
    save_json(exp_paths["metrics"] / "extract_summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description="MRM entry point")
    parser.add_argument("mode", choices=["train", "eval", "extract"])
    parser.add_argument("--config", default="configs/mrm.yaml")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    if "modality_names" not in cfg:
        cfg["modality_names"] = MODALITIES

    if args.mode == "train":
        run_train(cfg)
    elif args.mode == "eval":
        run_eval(cfg)
    elif args.mode == "extract":
        run_extract(cfg)


if __name__ == "__main__":
    main()
