"""RMVPE pitch extractor for F0 cache (Hz → resized to StyleTTS2/Kokoro mel frames)."""
from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
RVC_ROOT = ROOT / "rvc"
DEFAULT_RMVPE_CKPT = ROOT / "timbre_encoder" / "checkpoints" / "rmvpe" / "rmvpe.pt"
FALLBACK_CKPTS = (
    DEFAULT_RMVPE_CKPT,
    RVC_ROOT / "assets" / "rmvpe" / "rmvpe.pt",
    ROOT.parent / "rvc" / "assets" / "rmvpe" / "rmvpe.pt",
)

RMVPE_SR = 16000
RMVPE_HOP = 160  # 100 fps at 16 kHz


def resolve_rmvpe_ckpt(explicit: str | Path | None = None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f"RMVPE checkpoint not found: {p}")
        return p
    for p in FALLBACK_CKPTS:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "RMVPE checkpoint not found. Place rmvpe.pt at "
        f"{DEFAULT_RMVPE_CKPT} (or rvc/assets/rmvpe/rmvpe.pt)."
    )


def load_rmvpe(
    device: torch.device,
    model_path: str | Path | None = None,
    is_half: bool | None = None,
):
    """Load RVC RMVPE (use_jit=False). Returns (model, ckpt_path)."""
    ckpt = resolve_rmvpe_ckpt(model_path)
    if str(RVC_ROOT) not in sys.path:
        sys.path.insert(0, str(RVC_ROOT))
    from infer.lib.rmvpe import RMVPE  # noqa: WPS433

    if is_half is None:
        is_half = device.type == "cuda"
    print(f"loading RMVPE from {ckpt} half={is_half} device={device}", flush=True)
    model = RMVPE(str(ckpt), is_half=is_half, device=str(device), use_jit=False)
    return model, ckpt


def resize_f0(f0: np.ndarray | torch.Tensor, n_frames: int) -> torch.Tensor:
    """Linear-resample F0 curve to n_frames; keep unvoiced (~0) near zero."""
    if isinstance(f0, torch.Tensor):
        x = f0.float().view(-1).cpu()
    else:
        x = torch.from_numpy(np.asarray(f0, dtype=np.float32)).view(-1)
    if x.numel() == n_frames:
        return x
    if x.numel() < 2 or n_frames < 1:
        return torch.zeros(n_frames, dtype=torch.float32)
    y = F.interpolate(x.view(1, 1, -1), size=n_frames, mode="linear", align_corners=True)
    return y.view(-1).float()


@torch.no_grad()
def extract_f0_rmvpe(
    rmvpe,
    wav_24k: np.ndarray,
    n_mel_frames: int,
    thred: float = 0.01,
    smooth: bool = False,
    max_uv_gap: int = 16,
    median_k: int = 7,
    edge_fade: int = 4,
) -> torch.Tensor:
    """
    wav @ 24 kHz (same trim/cap as f0n cache) → F0 [n_mel_frames] Hz float32 CPU.
    Internally resamples to 16 kHz for RMVPE (hop=160).

    Defaults lean sensitive: low ``thred`` (fewer hard UV cuts), larger UV-gap fill.
    """
    if wav_24k.ndim > 1:
        wav_24k = np.asarray(wav_24k, dtype=np.float32).reshape(-1)
    audio_16k = librosa.resample(
        wav_24k.astype(np.float32),
        orig_sr=24000,
        target_sr=RMVPE_SR,
    )
    f0 = rmvpe.infer_from_audio(audio_16k, thred=thred)
    out = resize_f0(f0, n_mel_frames)
    if smooth:
        out = smooth_f0(
            out, max_uv_gap=max_uv_gap, median_k=median_k, edge_fade=edge_fade
        )
    return out


def _fill_short_uv_gaps(f0: np.ndarray, max_gap: int, voiced_thr: float = 1.0) -> np.ndarray:
    """Linear-fill unvoiced runs of length <= max_gap when bordered by voiced frames."""
    y = f0.astype(np.float64).copy()
    n = y.size
    if n < 3 or max_gap < 1:
        return y.astype(np.float32)
    i = 0
    while i < n:
        if y[i] > voiced_thr:
            i += 1
            continue
        j = i
        while j < n and y[j] <= voiced_thr:
            j += 1
        gap = j - i
        left = y[i - 1] if i > 0 else 0.0
        right = y[j] if j < n else 0.0
        if gap <= max_gap and left > voiced_thr and right > voiced_thr:
            for k in range(gap):
                a = (k + 1) / (gap + 1)
                y[i + k] = left * (1.0 - a) + right * a
        i = j
    return y.astype(np.float32)


def _median_voiced(f0: np.ndarray, k: int, voiced_thr: float = 1.0) -> np.ndarray:
    """Odd-window median; unvoiced frames stay 0; edges use available neighbors."""
    if k < 3 or k % 2 == 0:
        return f0
    y = f0.astype(np.float64).copy()
    n = y.size
    r = k // 2
    out = y.copy()
    for i in range(n):
        if y[i] <= voiced_thr:
            continue
        lo, hi = max(0, i - r), min(n, i + r + 1)
        win = y[lo:hi]
        win = win[win > voiced_thr]
        if win.size:
            out[i] = float(np.median(win))
    return out.astype(np.float32)


def _soften_uv_edges(f0: np.ndarray, fade: int = 4, voiced_thr: float = 1.0) -> np.ndarray:
    """Ramp F0 at start/end of each voiced run so UV cuts are less cliff-like."""
    if fade < 1:
        return f0
    y = f0.astype(np.float64).copy()
    n = y.size
    voiced = y > voiced_thr
    i = 0
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        j = i
        while j < n and voiced[j]:
            j += 1
        run = j - i
        f = min(fade, max(1, run // 3))
        for t in range(f):
            w_in = (t + 1) / (f + 1)
            y[i + t] *= w_in
        for t in range(f):
            w_out = 1.0 - (t + 1) / (f + 1)
            y[j - f + t] *= w_out
        i = j
    return y.astype(np.float32)


def smooth_f0(
    f0: np.ndarray | torch.Tensor,
    max_uv_gap: int = 16,
    median_k: int = 7,
    voiced_thr: float = 1.0,
    edge_fade: int = 4,
) -> torch.Tensor:
    """
    Soften RMVPE cliffs: fill short UV holes, median-filter voiced frames,
    optional edge fade so drops are less vertical.
    """
    if isinstance(f0, torch.Tensor):
        arr = f0.detach().float().view(-1).cpu().numpy()
    else:
        arr = np.asarray(f0, dtype=np.float32).reshape(-1)
    arr = _fill_short_uv_gaps(arr, max_gap=max_uv_gap, voiced_thr=voiced_thr)
    arr = _median_voiced(arr, k=median_k, voiced_thr=voiced_thr)
    arr = _soften_uv_edges(arr, fade=edge_fade, voiced_thr=voiced_thr)
    return torch.from_numpy(arr)
