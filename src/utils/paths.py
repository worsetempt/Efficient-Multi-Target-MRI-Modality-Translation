from __future__ import annotations

from pathlib import Path


def build_experiment_paths(output_root: str | Path, experiment_name: str) -> dict:
    root = Path(output_root) / experiment_name
    paths = {
        "root": root,
        "checkpoints": root / "checkpoints",
        "metrics": root / "metrics",
        "visuals": root / "visuals",
        "artifacts": root / "artifacts",
        "latents": root / "artifacts" / "latents",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths
