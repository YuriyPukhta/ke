"""Fine-tune StyleTTS2 TMA (ASRCNN) so soft alignment matches Kokoro duration_proj."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PACKAGE_ROOT / "data"
DEFAULT_CKPT_DIR = PACKAGE_ROOT / "checkpoints"
DEFAULT_RESULTS_DIR = PACKAGE_ROOT / "results"

__all__ = [
    "PACKAGE_ROOT",
    "DEFAULT_DATA_DIR",
    "DEFAULT_CKPT_DIR",
    "DEFAULT_RESULTS_DIR",
]
