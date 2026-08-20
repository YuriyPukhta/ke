#!/usr/bin/env python3
"""Preflight checks before RunPod training."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from runpod_training.paths import STYLETTS2_ROOT, ensure_styletts2_on_path, ensure_volume, volume_layout


def check_file(path: Path, label: str, errors: list[str]) -> None:
    if not path.is_file():
        errors.append(f"missing {label}: {path}")


def check_dir(path: Path, label: str, errors: list[str]) -> None:
    if not path.is_dir():
        errors.append(f"missing {label}: {path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    errors: list[str] = []
    warns: list[str] = []

    try:
        ensure_styletts2_on_path()
    except FileNotFoundError as exc:
        errors.append(str(exc))

    layout = ensure_volume(Path(cfg.get("run_root", "/workspace/kokoro-run")))
    data = cfg["data_params"]
    check_dir(Path(data["root_path"]), "LibriTTS_R root", errors)
    check_file(Path(data["train_index"]), "train index", errors)
    check_file(Path(data["val_index"]), "val index", errors)
    check_file(Path(cfg["pretrained_model"]), "kokoro_base.pth", errors)

    st2 = STYLETTS2_ROOT
    for key, label in [("ASR_path", "ASR"), ("ASR_config", "ASR config"), ("F0_path", "JDC"), ("PLBERT_dir", "PLBERT")]:
        raw = Path(cfg[key])
        path = raw if raw.exists() else st2 / raw
        if key.endswith("dir"):
            check_dir(path, label, errors)
        else:
            check_file(path, label, errors)

    style_ckpt = cfg.get("style_init", {}).get("kikiri_stage1_checkpoint")
    if style_ckpt and not Path(style_ckpt).is_file():
        warns.append(f"style_init checkpoint missing (will random-init shared): {style_ckpt}")

    if not os.environ.get("WANDB_API_KEY") and cfg.get("wandb", True):
        warns.append("WANDB_API_KEY unset; TensorBoard/JSONL only")

    # GPU
    try:
        import torch

        if not torch.cuda.is_available():
            warns.append("CUDA not available")
        else:
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU: {name} ({mem:.1f} GiB)")
    except Exception as exc:
        warns.append(f"torch check failed: {exc}")

    report = {
        "ok": not errors,
        "errors": errors,
        "warnings": warns,
        "volume": {k: str(v) for k, v in layout.items()},
    }
    print(json.dumps(report, indent=2))
    (layout["state"] / "preflight.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
