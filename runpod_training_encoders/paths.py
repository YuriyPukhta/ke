"""Resolve repository and StyleTTS2 dependency paths without modifying third_party."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent
KIKIRI_ROOT = REPO_ROOT / "third_party" / "kikiri-tts"
STYLETTS2_ROOT = KIKIRI_ROOT / "StyleTTS2"
DEFAULT_VOLUME = Path("/workspace/kokoro-run")


def ensure_styletts2_on_path() -> Path:
    """Put kikiri StyleTTS2 on sys.path for imports (models, losses, utils)."""
    target = STYLETTS2_ROOT
    if not target.is_dir():
        fallback = Path("/workspace/kokoro/third_party/kikiri-tts/StyleTTS2")
        if fallback.is_dir():
            target = fallback
        else:
            raise FileNotFoundError(
                f"StyleTTS2 dependency missing: {STYLETTS2_ROOT} or {fallback}. "
                "Init kikiri-tts submodules under third_party/kikiri-tts."
            )
    root = str(target)
    if root not in sys.path:
        sys.path.insert(0, root)
    return target


def volume_layout(run_root: Path | None = None) -> dict[str, Path]:
    root = Path(run_root) if run_root else DEFAULT_VOLUME
    layout = {
        "root": root,
        "checkpoints": root / "checkpoints",
        "stage1": root / "checkpoints" / "stage1",
        "stage2": root / "checkpoints" / "stage2",
        "logs": root / "logs" / "tensorboard",
        "metrics": root / "metrics",
        "samples": root / "samples",
        "exports": root / "exports",
        "kmodel": root / "exports" / "kmodel",
        "voicepacks": root / "exports" / "voicepacks",
        "configs": root / "configs",
        "state": root / "state",
        "indexes": root / "indexes",
        "weights": root / "weights",
    }
    return layout


def ensure_volume(run_root: Path | None = None) -> dict[str, Path]:
    layout = volume_layout(run_root)
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout
