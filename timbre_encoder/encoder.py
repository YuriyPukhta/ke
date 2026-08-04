"""
Timbre encoder for Kokoro: StyleTTS2 StyleEncoder -> 128-D vector (ref_s[:, :128]).

Architecture matches yl4579/StyleTTS2 models.py StyleEncoder.
Use mel_preprocess (NOT optimize_ref_s mel params).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

from .mel_preprocess import load_audio_mel, waveform_to_mel


class _LearnedDownSample(nn.Module):
    def __init__(self, layer_type: str, dim_in: int):
        super().__init__()
        if layer_type == "none":
            self.conv = nn.Identity()
        elif layer_type == "timepreserve":
            self.conv = spectral_norm(
                nn.Conv2d(dim_in, dim_in, kernel_size=(3, 1), stride=(2, 1), groups=dim_in, padding=(1, 0))
            )
        elif layer_type == "half":
            self.conv = spectral_norm(
                nn.Conv2d(dim_in, dim_in, kernel_size=(3, 3), stride=(2, 2), groups=dim_in, padding=1)
            )
        else:
            raise ValueError(f"Unknown downsample type: {layer_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _DownSample(nn.Module):
    def __init__(self, layer_type: str):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layer_type == "none":
            return x
        if self.layer_type == "timepreserve":
            return F.avg_pool2d(x, (2, 1))
        if self.layer_type == "half":
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool2d(x, 2)
        raise ValueError(f"Unknown downsample type: {self.layer_type}")


class _ResBlk(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, downsample: str = "none"):
        super().__init__()
        self.actv = nn.LeakyReLU(0.2)
        self.downsample = _DownSample(downsample)
        self.downsample_res = _LearnedDownSample(downsample, dim_in)
        self.learned_sc = dim_in != dim_out
        self.conv1 = spectral_norm(nn.Conv2d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(dim_in, dim_out, 3, 1, 1))
        self.conv1x1 = (
            spectral_norm(nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)) if self.learned_sc else None
        )

    def _shortcut(self, x: torch.Tensor) -> torch.Tensor:
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _residual(self, x: torch.Tensor) -> torch.Tensor:
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample_res(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self._shortcut(x) + self._residual(x)) / math.sqrt(2)


class StyleEncoder(nn.Module):
    """StyleTTS2 acoustic style encoder: mel image [B,1,80,T] -> [B, style_dim]."""

    def __init__(self, dim_in: int = 64, style_dim: int = 128, max_conv_dim: int = 512):
        super().__init__()
        blocks: list[nn.Module] = [spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1))]
        repeat_num = 4
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks.append(_ResBlk(dim_in, dim_out, downsample="half"))
            dim_in = dim_out
        blocks += [
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(dim_out, dim_out, 5, 1, 0)),
            nn.AdaptiveAvgPool2d(1),
            nn.LeakyReLU(0.2),
        ]
        self.shared = nn.Sequential(*blocks)
        self.unshared = nn.Linear(dim_out, style_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        return self.unshared(h)


@dataclass
class TimbreEncoderConfig:
    dim_in: int = 64
    style_dim: int = 128
    max_conv_dim: int = 512


class TimbreEncoder(nn.Module):
    """
    Reference audio -> Kokoro timbre vector ref_s[:, :128].

    Inference:
        enc = TimbreEncoder()
        timbre = enc.encode_wav("reference.wav")  # [1, 128]
    """

    def __init__(self, config: TimbreEncoderConfig | None = None):
        super().__init__()
        config = config or TimbreEncoderConfig()
        self.config = config
        self.encoder = StyleEncoder(
            dim_in=config.dim_in,
            style_dim=config.style_dim,
            max_conv_dim=config.max_conv_dim,
        )

    def encode_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """mel [B, 80, T] -> timbre [B, 128]."""
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        x = mel.unsqueeze(1)
        return self.encoder(x)

    def encode_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """Waveform [T] or [B, T] @ 24 kHz -> [B, 128]."""
        mel = waveform_to_mel(waveform, device=waveform.device)
        return self.encode_mel(mel)

    @torch.no_grad()
    def encode_wav(self, path: str, device: torch.device | None = None) -> torch.Tensor:
        """Audio file -> timbre [1, 128]."""
        device = device or torch.device("cpu")
        self.eval().to(device)
        mel = load_audio_mel(path, device=device)
        return self.encode_mel(mel)
