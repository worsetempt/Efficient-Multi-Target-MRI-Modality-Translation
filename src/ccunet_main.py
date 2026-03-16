from __future__ import annotations

import argparse

from src.inference.ccunet_infer import run_inference
from src.training.ccunet_trainer import CCUNetConfig, CCUNetTrainer
from src.utils.config import load_yaml_config
from src.utils.io import save_json
from src.utils.paths import build_experiment_paths
from src.utils.reproducibility import configure_torch, resolve_device, set_seed



def build_train_config(cfg: dict) -> CCUNetConfig:
    tr = cfg["training"]
    return CCUNetConfig(
        train_image_dir=cfg["data"]["train_image_dir"],
        train_mrm_latent_path=cfg["data"]["train_mrm_latent_path"],
        modality_names=list(cfg["modality_names"]),
        output_root=cfg["output_root"],
        experiment_name=cfg["experiment_name"],
        device=cfg.get("device", "cuda"),
        seed=int(cfg.get("seed", 42)),
        epochs=int(tr["epochs"]),
        batch_size=int(tr["batch_size"]),
        learning_rate=float(tr["learning_rate"]),
        num_workers=int(tr.get("num_workers", 4)),
        persistent_workers=bool(tr.get("persistent_workers", True)),
        image_size=int(tr.get("image_size", 128)),
        latent_dim=int(tr["latent_dim"]),
        mod_embed_dim=int(tr.get("mod_embed_dim", 16)),
        train_samples_per_class=int(tr.get("train_samples_per_class", 3000)),
        val_total_samples=int(tr.get("val_total_samples", 2000)),
        split_seed=int(tr.get("split_seed", 42)),
    )



def run_train(cfg: dict):
    train_cfg = build_train_config(cfg)
    set_seed(train_cfg.seed)
    device = resolve_device(train_cfg.device)
    configure_torch(device)
    exp_paths = build_experiment_paths(train_cfg.output_root, train_cfg.experiment_name)
    save_json(exp_paths["root"] / "config.json", cfg)
    trainer = CCUNetTrainer(train_cfg, exp_paths, device)
    trainer.train()



def _resolve_ckpt(exp_paths: dict, name: str) -> str:
    return str(exp_paths["checkpoints"] / ("best.pt" if name == "best" else "last.pt"))



def _run_eval_like(cfg: dict, mode: str):
    exp_paths = build_experiment_paths(cfg["output_root"], cfg["experiment_name"])
    stage_cfg = cfg["evaluation"] if mode == "eval" else cfg["inference"]
    device = resolve_device(cfg.get("device", "cuda"))
    configure_torch(device)
    ckpt_path = _resolve_ckpt(exp_paths, stage_cfg["checkpoint"])
    out_dir = exp_paths["visuals"] / ("eval" if mode == "eval" else "infer")
    summary = run_inference(
        ckpt_path=ckpt_path,
        infer_dir=cfg["data"]["eval_image_dir"],
        latent_path=cfg["data"]["infer_generated_latent_path"],
        modality_names=list(cfg["modality_names"]),
        source_modality=stage_cfg["source_modality"],
        target_modality=stage_cfg["target_modality"],
        auto_use_latent_metadata=bool(stage_cfg.get("auto_use_latent_metadata", True)),
        max_samples=int(stage_cfg.get("max_samples", 100)),
        output_dir=out_dir,
        device=device,
        save_visualizations=bool(stage_cfg.get("save_visualizations", True if mode == "eval" else True)),
    )
    save_json(exp_paths["metrics"] / f"{mode}.json", summary)



def main():
    parser = argparse.ArgumentParser(description="CCU-Net entry point")
    parser.add_argument("mode", choices=["train", "infer", "eval"])
    parser.add_argument("--config", default="configs/ccunet.yaml")
    args = parser.parse_args()
    cfg = load_yaml_config(args.config)
    if args.mode == "train":
        run_train(cfg)
    elif args.mode == "infer":
        _run_eval_like(cfg, mode="infer")
    else:
        _run_eval_like(cfg, mode="eval")


if __name__ == "__main__":
    main()
