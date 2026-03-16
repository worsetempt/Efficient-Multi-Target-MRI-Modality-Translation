from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.latent_translation_eval import compute_fd_metrics
from src.evaluation.latent_translation_visualization import plot_generated_umap
from src.inference.translate_latents import generate_translated_latents, save_translated_outputs
from src.training.latent_translator_trainer import TranslatorConfig, TranslatorTrainer
from src.utils.config import load_yaml_config
from src.utils.io import save_json
from src.utils.paths import build_experiment_paths
from src.utils.reproducibility import configure_torch, resolve_device, set_seed


def build_config(cfg: dict) -> TranslatorConfig:
    return TranslatorConfig(
        train_latent_paths=cfg["data"]["train_latent_paths"],
        val_latent_paths=cfg["data"]["val_latent_paths"],
        test_latent_paths=cfg["data"].get("test_latent_paths"),
        modality_names=list(cfg["modality_names"]),
        output_root=cfg["output_root"],
        experiment_name=cfg["experiment_name"],
        device=cfg.get("device", "cuda"),
        seed=int(cfg.get("seed", 42)),
        epochs=int(cfg["training"]["epochs"]),
        batch_size=int(cfg["training"]["batch_size"]),
        num_workers=int(cfg["training"].get("num_workers", 0)),
        learning_rate=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        grad_clip=float(cfg["training"]["grad_clip"]),
        amp=bool(cfg["training"]["amp"]),
        resume_ckpt=cfg["training"].get("resume_ckpt"),
        save_every=int(cfg["training"]["save_every"]),
        ema_decay=float(cfg["training"]["ema_decay"]),
        hidden_dim=int(cfg["training"]["hidden_dim"]),
        n_blocks=int(cfg["training"]["n_blocks"]),
        dropout=float(cfg["training"]["dropout"]),
        use_layernorm=bool(cfg["training"]["use_layernorm"]),
        lambda_l1=float(cfg["training"]["lambda_l1"]),
        lambda_cos=float(cfg["training"]["lambda_cos"]),
        lambda_delta=float(cfg["training"]["lambda_delta"]),
        lambda_pair_consistency=float(cfg["training"]["lambda_pair_consistency"]),
        lambda_centroid=float(cfg["training"]["lambda_centroid"]),
        max_translate_samples=cfg["generation"].get("max_samples"),
    )


def build_system(cfg: dict):
    tr_cfg = build_config(cfg)
    set_seed(tr_cfg.seed)
    device = resolve_device(tr_cfg.device)
    configure_torch(device)
    exp_paths = build_experiment_paths(tr_cfg.output_root, tr_cfg.experiment_name)
    save_json(exp_paths["root"] / "config.json", cfg)
    trainer = TranslatorTrainer(tr_cfg, exp_paths, device)
    return trainer, exp_paths


def _resolve_ckpt(exp_paths: dict, name: str) -> str:
    return str(exp_paths["checkpoints"] / ("best.pt" if name == "best" else "last.pt"))


def run_train(cfg: dict):
    trainer, _ = build_system(cfg)
    trainer.train()


def run_generate(cfg: dict):
    trainer, exp_paths = build_system(cfg)
    gen_cfg = cfg["generation"]
    ckpt_path = _resolve_ckpt(exp_paths, gen_cfg["checkpoint"])
    outputs = generate_translated_latents(trainer, split=gen_cfg["split"], ckpt_path=ckpt_path, use_ema=bool(gen_cfg["use_ema"]))
    save_translated_outputs(outputs, trainer, gen_cfg["split"], exp_paths["artifacts"] / "generated_latents")


def run_eval(cfg: dict):
    trainer, exp_paths = build_system(cfg)
    eval_cfg = cfg["evaluation"]
    gen_root = Path(eval_cfg.get("generated_root") or (exp_paths["artifacts"] / "generated_latents"))
    ckpt_path = _resolve_ckpt(exp_paths, eval_cfg["checkpoint"])
    if not gen_root.exists() or not list(gen_root.glob("*_to_*.pt")):
        outputs = generate_translated_latents(trainer, split=eval_cfg["split"], ckpt_path=ckpt_path, use_ema=bool(eval_cfg["use_ema"]))
        save_translated_outputs(outputs, trainer, eval_cfg["split"], gen_root)
    fd_metrics = compute_fd_metrics(cfg["data"][f"{eval_cfg['split']}_latent_paths"], gen_root, int(eval_cfg["fd_max_samples"]))
    payload = {"fd_metrics": fd_metrics}
    if eval_cfg["umap"]["enabled"]:
        payload["umap_path"] = plot_generated_umap(
            cfg["data"][f"{eval_cfg['split']}_latent_paths"],
            gen_root,
            exp_paths["visuals"] / "generated_umap.png",
            max_samples=int(eval_cfg["max_samples"]),
            n_neighbors=int(eval_cfg["umap"]["n_neighbors"]),
            min_dist=float(eval_cfg["umap"]["min_dist"]),
        )
    save_json(exp_paths["metrics"] / "eval.json", payload)


def main():
    parser = argparse.ArgumentParser(description="Latent translator entry point")
    parser.add_argument("mode", choices=["train", "generate", "eval"])
    parser.add_argument("--config", default="configs/latent_translator.yaml")
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
