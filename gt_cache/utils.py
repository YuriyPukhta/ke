"""Shared helpers for caching F0 / N / d_gt-related curves on LibriTTS."""
from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
ST2_ROOT = ROOT / "third_party" / "StyleTTS2"
CKPT_DIR = ROOT / "timbre_encoder" / "checkpoints" / "styletts2_libritts" / "Models" / "LibriTTS"
DEFAULT_FEATURES_ROOT = Path(__file__).resolve().parent / "features"
DEFAULT_MANIFEST_DIR = ROOT / "timbre_encoder" / "manifests"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party"))
sys.path.insert(0, str(ST2_ROOT))

from timbre_encoder.mel_preprocess import SAMPLE_RATE, waveform_to_mel  # noqa: E402
from timbre_encoder.st2_runtime import patch_torch_load  # noqa: E402

F0_FIELDS = ("f0_status", "f0_len")
N_FIELDS = ("n_status", "n_len")
F0_RMVPE_FIELDS = ("f0_rmvpe_status", "f0_rmvpe_len")
F0_JDC_FT_FIELDS = ("f0_jdc_ft_status", "f0_jdc_ft_len")
FEATURE_KINDS = ("f0", "n", "f0_rmvpe", "f0_jdc_ft")


def audio_rel_under_libritts(audio_path: str | Path) -> Path:
    """LibriTTS_R/.../utt.wav → train-clean-100/.../utt.npy relative."""
    p = Path(audio_path)
    parts = p.parts
    if "LibriTTS_R" in parts:
        i = parts.index("LibriTTS_R")
        return Path(*parts[i + 1 :]).with_suffix(".npy")
    return Path(p.parent.name) / f"{p.stem}.npy"


def feature_path(audio_path: str | Path, kind: str, features_root: Path | str) -> Path:
    assert kind in FEATURE_KINDS, f"unknown feature kind: {kind}"
    return Path(features_root) / kind / audio_rel_under_libritts(audio_path)


def load_wav_for_cache(
    path: str,
    max_wav_sec: float = 8.0,
    trim_top_db: int = 30,
) -> np.ndarray | None:
    """Same trim/cap as LibriTTSDataset / cache_d_gt."""
    if not path or not Path(path).is_file():
        return None
    wav, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    wav, _ = librosa.effects.trim(wav, top_db=trim_top_db)
    max_samples = int(max_wav_sec * SAMPLE_RATE)
    if wav.shape[0] > max_samples:
        wav = wav[:max_samples]
    if wav.shape[0] < SAMPLE_RATE // 4:
        return None
    return wav.astype(np.float32)


def wav_to_mel(wav: np.ndarray, device: torch.device) -> torch.Tensor:
    """[T] → mel [80, T_mel] on device."""
    wave_t = torch.from_numpy(wav)
    mel = waveform_to_mel(wave_t, device=device)
    return mel.squeeze(0)


def load_pitch_extractor(device: torch.device, weights: str | Path | None = None):
    """Load JDCNet; default = StyleTTS2 bst.t7, or a fine-tuned .t7 checkpoint."""
    patch_torch_load()
    from models import load_F0_models

    if weights:
        f0_path = Path(weights)
    else:
        config = yaml.safe_load(open(CKPT_DIR / "config.yml", encoding="utf-8"))
        f0_path = ST2_ROOT / config["F0_path"]
    if not f0_path.is_file():
        raise FileNotFoundError(f"JDC weights not found: {f0_path}")
    print(f"loading JDCNet from {f0_path}", flush=True)
    pe = load_F0_models(str(f0_path))
    pe.eval().to(device)
    for p in pe.parameters():
        p.requires_grad_(False)
    return pe


@torch.no_grad()
def extract_f0(pitch_extractor, mel: torch.Tensor) -> torch.Tensor:
    """mel [80, T] → F0 [T] float32 CPU."""
    F0_real, _, _ = pitch_extractor(mel.unsqueeze(0).unsqueeze(0))
    f0 = F0_real.float().squeeze().view(-1).cpu()
    return f0


@torch.no_grad()
def extract_n(mel: torch.Tensor) -> torch.Tensor:
    """mel [80, T] → N / log_norm [T] float32 CPU."""
    from utils import log_norm

    n = log_norm(mel.unsqueeze(0).unsqueeze(0)).squeeze().view(-1).cpu()
    return n.float()


def save_curve(path: Path, curve: torch.Tensor | np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = curve.detach().cpu().numpy() if isinstance(curve, torch.Tensor) else np.asarray(curve)
    np.save(path, arr.astype(np.float32))


def load_curve(path: Path | str) -> torch.Tensor | None:
    p = Path(path)
    if not p.is_file():
        return None
    return torch.from_numpy(np.load(p).astype(np.float32))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.stem + "_", suffix=".csv", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with tmp_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def ensure_fields(fieldnames: list[str], extra: tuple[str, ...]) -> list[str]:
    out = list(fieldnames)
    for col in extra:
        if col not in out:
            out.append(col)
    return out
