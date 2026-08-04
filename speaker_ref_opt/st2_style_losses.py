"""StyleTTS2 finetune losses usable with Kokoro waveform / F0 / N heads.

Ported from third_party/StyleTTS2/losses.py (MultiResolutionSTFT).
GAN / diffusion / TMA terms are omitted — Kokoro is frozen + forced d_gt.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class SpectralConvergenceLoss(nn.Module):
    def forward(self, x_mag: torch.Tensor, y_mag: torch.Tensor) -> torch.Tensor:
        return torch.norm(y_mag - x_mag, p=1) / (torch.norm(y_mag, p=1) + 1e-7)


class STFTLoss(nn.Module):
    """Log-mel spectral convergence (StyleTTS2 STFTLoss)."""

    def __init__(
        self,
        fft_size: int = 1024,
        shift_size: int = 120,
        win_length: int = 600,
        sample_rate: int = 24000,
        window=torch.hann_window,
    ):
        super().__init__()
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=fft_size,
            win_length=win_length,
            hop_length=shift_size,
            window_fn=window,
        )
        self.spectral_convergence_loss = SpectralConvergenceLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mean, std = -4.0, 4.0
        x_mag = (torch.log(1e-5 + self.to_mel(x)) - mean) / std
        y_mag = (torch.log(1e-5 + self.to_mel(y)) - mean) / std
        return self.spectral_convergence_loss(x_mag, y_mag)


class MultiResolutionSTFTLoss(nn.Module):
    """StyleTTS2 multi-resolution mel spectral loss (config_ft lambda_mel)."""

    def __init__(
        self,
        fft_sizes: list[int] | None = None,
        hop_sizes: list[int] | None = None,
        win_lengths: list[int] | None = None,
        sample_rate: int = 24000,
    ):
        super().__init__()
        fft_sizes = fft_sizes or [1024, 2048, 512]
        hop_sizes = hop_sizes or [120, 240, 50]
        win_lengths = win_lengths or [600, 1200, 240]
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = nn.ModuleList(
            [
                STFTLoss(fs, ss, wl, sample_rate=sample_rate)
                for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths)
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)
        n = min(x.shape[-1], y.shape[-1])
        x, y = x[..., :n], y[..., :n]
        loss = x.new_zeros(())
        for f in self.stft_losses:
            loss = loss + f(x, y)
        return loss / len(self.stft_losses)


def st2_f0_loss(f0_pred: torch.Tensor, f0_tgt: torch.Tensor) -> torch.Tensor:
    """StyleTTS2: smooth_l1(F0_real, F0_fake) / 10."""
    n = min(f0_pred.numel(), f0_tgt.numel())
    if n < 4:
        return f0_pred.new_zeros(())
    return F.smooth_l1_loss(f0_pred.view(-1)[:n], f0_tgt.view(-1)[:n].detach()) / 10.0


def st2_norm_loss(n_pred: torch.Tensor, n_tgt: torch.Tensor) -> torch.Tensor:
    """StyleTTS2: smooth_l1(N_real, N_fake)."""
    n = min(n_pred.numel(), n_tgt.numel())
    if n < 4:
        return n_pred.new_zeros(())
    return F.smooth_l1_loss(n_pred.view(-1)[:n], n_tgt.view(-1)[:n].detach())
