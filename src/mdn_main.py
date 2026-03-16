from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.mdn_metrics import compute_fd_metrics
from src.evaluation.mdn_visualization import plot_generated_umap
from src.inference.mdn_generate import save_generated_outputs
from src.training.mdn_trainer import DiffusionTrainer, MDNConfig
from src.utils.config import load_yaml_config
from src.utils.io import save_json
from src.utils.paths import build_experiment_paths
from src.utils.reproducibility import configure_torch, resolve_device, set_seed


def build_config(cfg: dict) -> MDNConfig:
    return MDNConfig(
        train_latent_path=cfg["data"]["train_latent_path"],
        eval_latent_path=cfg["data"]["eval_latent_path"],
        modality_names=list(cfg["modality_names"]),
        output_root=cfg["output_root"],
        experiment_name=cfg["experiment_name"],
        device=cfg.get("device", "cuda"),
        seed=int(cfg.get("seed", 42)),
        epochs=int(cfg["training"]["epochs"]),
        batch_size=int(cfg["training"]["batch_size"]),
        learning_rate=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        hidden_dim=int(cfg["training"]["hidden_dim"]),
        max_time_steps=int(cfg["training"]["max_time_steps"]),
        n_blocks=int(cfg["training"]["n_blocks"]),
        dropout=float(cfg["training"]["dropout"]),
        class_dropout_prob=float(cfg["training"]["class_dropout_prob"]),
        ema_decay=float(cfg["training"]["ema_decay"]),
        grad_clip=float(cfg["training"]["grad_clip"]),
        amp=bool(cfg["training"]["amp"]),
        resume_ckpt=cfg["training"].get("resume_ckpt"),
        lambda_x0=float(cfg["training"]["lambda_x0"]),
        lambda_proto=float(cfg["training"]["lambda_proto"]),
        lambda_stats=float(cfg["training"]["lambda_stats"]),
        lambda_style_ortho=float(cfg["training"]["lambda_style_ortho"]),
        lambda_style_margin=float(cfg["training"]["lambda_style_margin"]),
        lambda_class_sep=float(cfg["training"]["lambda_class_sep"]),
        metrics_num_samples=int(cfg["training"]["metrics_num_samples"]),
    )


def build_system(cfg: dict):
    mdn_cfg = build_config(cfg)
    set_seed(mdn_cfg.seed)
    device = resolve_device(mdn_cfg.device)
    configure_torch(device)
    exp_paths = build_experiment_paths(mdn_cfg.output_root, mdn_cfg.experiment_name)
    save_json(exp_paths["root"] / "config.json", cfg)
    trainer = DiffusionTrainer(mdn_cfg, exp_paths, device)
    return trainer, exp_paths, device


def run_train(cfg: dict):
    trainer, exp_paths, _ = build_system(cfg)
    trainer.train()


def _resolve_ckpt(exp_paths: dict, name: str) -> str:
    return str(exp_paths["checkpoints"] / ("best.pt" if name == "best" else "last.pt"))


def run_generate(cfg: dict):
    trainer, exp_paths, _ = build_system(cfg)
    gen_cfg = cfg["generation"]
    target_classes = [cfg["modality_names"].index(m) for m in gen_cfg["target_modalities"]]
    ckpt_path = _resolve_ckpt(exp_paths, gen_cfg["checkpoint"])
    outputs = trainer.generate(
        target_classes=target_classes,
        n_samples_per_class=int(gen_cfg["n_samples_per_class"]),
        n_sampling_steps=int(gen_cfg["n_sampling_steps"]),
        cfg_scale=float(gen_cfg["cfg_scale"]),
        ckpt_path=ckpt_path,
        use_ema=bool(gen_cfg["use_ema"]),
    )
    save_generated_outputs(outputs, cfg["modality_names"], exp_paths["artifacts"] / "generated_latents")


def run_eval(cfg: dict):
    trainer, exp_paths, _ = build_system(cfg)
    eval_cfg = cfg["evaluation"]
    gen_root = eval_cfg.get("generated_root") or str(exp_paths["artifacts"] / "generated_latents")
    ckpt_path = _resolve_ckpt(exp_paths, eval_cfg["checkpoint"])
    if not Path(gen_root).exists():
        outputs = trainer.generate(
            target_classes=list(range(len(cfg["modality_names"]))),
            n_samples_per_class=int(cfg["generation"]["n_samples_per_class"]),
            n_sampling_steps=int(cfg["generation"]["n_sampling_steps"]),
            cfg_scale=float(cfg["generation"]["cfg_scale"]),
            ckpt_path=ckpt_path,
            use_ema=bool(cfg["generation"]["use_ema"]),
        )
        save_generated_outputs(outputs, cfg["modality_names"], gen_root)
    metrics = compute_fd_metrics(cfg["data"]["eval_latent_path"], gen_root, cfg["modality_names"], int(eval_cfg["fd_max_samples"]))
    payload = {"fd_metrics": metrics}
    if eval_cfg["umap"]["enabled"]:
        payload["umap_path"] = plot_generated_umap(
            cfg["data"]["eval_latent_path"],
            gen_root,
            cfg["modality_names"],
            exp_paths["visuals"] / "generated_umap.png",
            max_samples=int(eval_cfg["max_samples"]),
            n_neighbors=int(eval_cfg["umap"]["n_neighbors"]),
            min_dist=float(eval_cfg["umap"]["min_dist"]),
        )
    save_json(exp_paths["metrics"] / "eval.json", payload)


def main():
    parser = argparse.ArgumentParser(description="MDN entry point")
    parser.add_argument("mode", choices=["train", "generate", "eval"])
    parser.add_argument("--config", default="configs/mdn.yaml")
    args = parser.parse_args()
    cfg = load_yaml_config(args.config)
    if args.mode == "train":
        run_train(cfg)
    elif args.mode == "generate":
        run_generate(cfg)
    else:
        run_eval(cfg)


if __name__ == "__main__":
    main()
