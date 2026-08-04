"""Mel preprocessing matching StyleTTS2 training (yl4579/StyleTTS2 meldataset.py)."""
from __future__ import annotations

import librosa
import numpy as np
import torch
import torchaudio

SAMPLE_RATE = 24000
N_FFT = 2048
WIN_LENGTH = 1200
HOP_LENGTH = 300
N_MELS = 80
MEL_MEAN = -4.0
MEL_STD = 4.0

_mel_transforms: dict[str, torchaudio.transforms.MelSpectrogram] = {}


def _get_mel_transform(device: torch.device) -> torchaudio.transforms.MelSpectrogram:
    key = f"{device.type}:{getattr(device, 'index', None)}"
    tr = _mel_transforms.get(key)
    if tr is None:
        tr = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_mels=N_MELS,
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
        ).to(device)
        _mel_transforms[key] = tr
    return tr


def waveform_to_mel(waveform: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
    """Waveform [T] or [B, T] -> mel [B, 80, T_mel] normalized like StyleTTS2."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    device = device or waveform.device
    waveform = waveform.to(device).float()
    mel = _get_mel_transform(device)(waveform)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    mel = (mel - MEL_MEAN) / MEL_STD
    t = mel.size(-1)
    if t % 2 == 1:
        mel = mel[..., :-1]
    return mel


def load_audio_mel(
    path: str,
    device: torch.device | None = None,
    trim_top_db: int = 30,
    pad_ms: float = 5000 / SAMPLE_RATE,
) -> torch.Tensor:
    """Load wav, resample to 24 kHz, trim silence, pad edges, return mel [1, 80, T]."""
    wave, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    wave, _ = librosa.effects.trim(wave, top_db=trim_top_db)
    pad = int(pad_ms * SAMPLE_RATE)
    wave = np.concatenate([np.zeros(pad), wave, np.zeros(pad)])
    device = device or torch.device("cpu")
    return waveform_to_mel(torch.from_numpy(wave), device=device)
