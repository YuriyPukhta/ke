"""Atomic checkpoint save / resume / retention for RunPod volumes."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _atomic_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def capture_rng() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: Path,
    model: dict,
    optimizer: Any | None,
    schedulers: dict | None,
    epoch: int,
    iters: int,
    phase: str,
    metrics: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    net = {k: v.state_dict() for k, v in model.items()}
    payload = {
        "net": net,
        "epoch": int(epoch),
        "iters": int(iters),
        "phase": phase,
        "metrics": metrics or {},
        "rng": capture_rng(),
        "extra": extra or {},
    }
    if optimizer is not None:
        if hasattr(optimizer, "state_dict"):
            payload["optimizer"] = optimizer.state_dict()
        elif isinstance(optimizer, dict):
            payload["optimizer"] = {k: opt.state_dict() for k, opt in optimizer.items()}
    if schedulers:
        payload["schedulers"] = {k: sch.state_dict() for k, sch in schedulers.items()}
    _atomic_torch_save(payload, path)
    logger.info("saved checkpoint %s (epoch=%s iters=%s)", path, epoch, iters)
    return path


def load_checkpoint(
    path: Path,
    model: dict,
    optimizer: Any | None = None,
    schedulers: dict | None = None,
    load_optimizer: bool = True,
    ignore_modules: list[str] | None = None,
) -> tuple[int, int, dict[str, Any]]:
    ignore = set(ignore_modules or [])
    state = torch.load(path, map_location="cpu", weights_only=False)
    params = state.get("net", state)
    for key in model:
        if key in params and key not in ignore:
            missing, unexpected = model[key].load_state_dict(params[key], strict=False)
            print(f"{key} loaded (missing={len(missing)} unexpected={len(unexpected)})")
    epoch = int(state.get("epoch", 0))
    iters = int(state.get("iters", 0))
    if load_optimizer and optimizer is not None and "optimizer" in state:
        try:
            optimizer.load_state_dict(state["optimizer"])
        except Exception as exc:
            logger.warning("optimizer restore failed: %s", exc)
    if load_optimizer and schedulers and "schedulers" in state:
        for key, sch in schedulers.items():
            if key in state["schedulers"]:
                try:
                    sch.load_state_dict(state["schedulers"][key])
                except Exception as exc:
                    logger.warning("scheduler %s restore failed: %s", key, exc)
    restore_rng(state.get("rng"))
    return epoch, iters, state


def update_latest_pointer(state_dir: Path, stage: str, path: Path, meta: dict[str, Any]) -> None:
    payload = {"stage": stage, "path": str(path), **meta}
    _atomic_json(payload, state_dir / "latest.json")
    _atomic_json(payload, state_dir / f"latest_{stage}.json")


def apply_retention(
    ckpt_dir: Path,
    keep_last: int = 3,
    protected_names: tuple[str, ...] = ("latest.pth", "best_mel.pth", "best_val.pth", "first_stage.pth"),
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(
        [p for p in ckpt_dir.glob("epoch_*.pth") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old in files[keep_last:]:
        if old.name in protected_names:
            continue
        try:
            old.unlink()
        except OSError:
            pass


def resolve_resume(path: str | Path | None, state_dir: Path, stage: str) -> Path | None:
    if path is None:
        return None
    if str(path) == "auto":
        pointer = state_dir / f"latest_{stage}.json"
        if not pointer.is_file():
            pointer = state_dir / "latest.json"
        if not pointer.is_file():
            return None
        data = json.loads(pointer.read_text(encoding="utf-8"))
        candidate = Path(data["path"])
        return candidate if candidate.is_file() else None
    p = Path(path)
    return p if p.is_file() else None
