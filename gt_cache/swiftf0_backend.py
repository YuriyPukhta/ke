"""SwiftF0 pitch extractor → resized to StyleTTS2/Kokoro mel frames."""
from __future__ import annotations

import numpy as np
import torch

from gt_cache.rmvpe_backend import resize_f0, smooth_f0

# SwiftF0 internal: 16 kHz, hop 256 → 62.5 fps
SWIFT_SR = 16000


def load_swiftf0(
    fmin: float = 65.0,
    fmax: float = 500.0,
    confidence_threshold: float = 0.55,
):
    from swift_f0 import SwiftF0

    print(
        f"loading SwiftF0 fmin={fmin} fmax={fmax} conf={confidence_threshold}",
        flush=True,
    )
    return SwiftF0(fmin=fmin, fmax=fmax, confidence_threshold=confidence_threshold)


def extract_f0_swift(
    detector,
    wav_24k: np.ndarray,
    n_mel_frames: int,
    sample_rate: int = 24000,
    smooth: bool = False,
    max_uv_gap: int = 16,
    median_k: int = 7,
    edge_fade: int = 4,
) -> torch.Tensor:
    """
    wav @ 24 kHz → F0 [n_mel_frames] Hz float32 CPU.
    Unvoiced frames forced to 0 via SwiftF0.voicing mask.
    """
    if wav_24k.ndim > 1:
        wav_24k = np.asarray(wav_24k, dtype=np.float32).reshape(-1)
    else:
        wav_24k = np.asarray(wav_24k, dtype=np.float32)

    result = detector.detect_from_array(wav_24k, sample_rate)
    f0 = np.asarray(result.pitch_hz, dtype=np.float32).copy()
    voiced = np.asarray(result.voicing, dtype=bool)
    f0[~voiced] = 0.0

    out = resize_f0(f0, n_mel_frames)
    if smooth:
        out = smooth_f0(
            out, max_uv_gap=max_uv_gap, median_k=median_k, edge_fade=edge_fade
        )
    return out
