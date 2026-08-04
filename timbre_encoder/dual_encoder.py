"""Dual StyleTTS2-style encoders: acoustic s_a[:128] + prosodic s_p[128:].

Outputs are **residuals** on a Kokoro baseline voice so AdaIN stays in-distribution:
  s = baseline + scale * tanh(delta)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .encoder import StyleEncoder
from .mel_preprocess import load_audio_mel, waveform_to_mel


@dataclass
class DualStyleEncoderConfig:
    dim_in: int = 64
    style_dim: int = 128
    max_conv_dim: int = 512
    residual_scale: float = 0.25


class DualStyleEncoder(nn.Module):
    """
    mel [B, 80, T] -> deltas; apply_baseline() -> (s_a, s_p) in Kokoro space.

    Last linear layers are zero-initialized so training starts ≈ baseline voice.
    """

    def __init__(self, config: DualStyleEncoderConfig | None = None):
        super().__init__()
        config = config or DualStyleEncoderConfig()
        self.config = config
        kw = dict(
            dim_in=config.dim_in,
            style_dim=config.style_dim,
            max_conv_dim=config.max_conv_dim,
        )
        self.acoustic = StyleEncoder(**kw)  # E_a -> delta_a
        self.prosodic = StyleEncoder(**kw)  # E_p -> delta_p
        self.residual_scale = float(config.residual_scale)
        self._zero_init_heads()

    def _zero_init_heads(self) -> None:
        for enc in (self.acoustic, self.prosodic):
            nn.init.zeros_(enc.unshared.weight)
            nn.init.zeros_(enc.unshared.bias)

    def forward_deltas(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if mel.dim() == 3:
            x = mel.unsqueeze(1)
        elif mel.dim() == 4:
            x = mel
        else:
            raise ValueError(f"Expected mel [B,80,T] or [B,1,80,T], got {tuple(mel.shape)}")
        return self.acoustic(x), self.prosodic(x)

    def apply_baseline(
        self,
        delta_a: torch.Tensor,
        delta_p: torch.Tensor,
        baseline_ref_s: torch.Tensor,
        timbre_only: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        baseline_ref_s: [1, 256] or [B, 256] Kokoro style vector (e.g. af_heart slot).
        """
        if baseline_ref_s.dim() == 1:
            baseline_ref_s = baseline_ref_s.unsqueeze(0)
        if baseline_ref_s.shape[0] == 1 and delta_a.shape[0] > 1:
            baseline_ref_s = baseline_ref_s.expand(delta_a.shape[0], -1)
        b_a = baseline_ref_s[:, :128]
        b_p = baseline_ref_s[:, 128:]
        s_a = b_a + self.residual_scale * torch.tanh(delta_a)
        if timbre_only:
            s_p = b_p
        else:
            s_p = b_p + self.residual_scale * torch.tanh(delta_p)
        return s_a, s_p

    def forward(
        self,
        mel: torch.Tensor,
        baseline_ref_s: torch.Tensor,
        timbre_only: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        da, dp = self.forward_deltas(mel)
        return self.apply_baseline(da, dp, baseline_ref_s, timbre_only=timbre_only)

    def encode_ref_s(
        self,
        mel: torch.Tensor,
        baseline_ref_s: torch.Tensor,
        timbre_only: bool = True,
    ) -> torch.Tensor:
        s_a, s_p = self.forward(mel, baseline_ref_s, timbre_only=timbre_only)
        return torch.cat([s_a, s_p], dim=-1)

    @torch.no_grad()
    def encode_wav(
        self,
        path: str,
        baseline_ref_s: torch.Tensor,
        device: torch.device | None = None,
        timbre_only: bool = True,
    ) -> torch.Tensor:
        device = device or torch.device("cpu")
        self.eval().to(device)
        mel = load_audio_mel(path, device=device)
        return self.encode_ref_s(
            mel, baseline_ref_s.to(device), timbre_only=timbre_only
        )


# StyleEncoder: 4× half-pool then Conv2d 5×5 (no pad). Need T/16 >= 5 → T >= 80.
MIN_STYLE_FRAMES = 80


def crop_mel_for_style(
    mel: torch.Tensor,
    max_frames: int = 192,
    min_frames: int = MIN_STYLE_FRAMES,
) -> torch.Tensor:
    """
    Random (train) / center (eval) crop along time for style encoders.
    Pads short mels so the 5×5 head after 4 half-pools does not crash (needs T>=80).
    """
    if mel.dim() == 2:
        mel = mel.unsqueeze(0)
    b, n_mels, t = mel.shape
    if t < min_frames:
        mel = torch.nn.functional.pad(mel, (0, min_frames - t))
        t = mel.shape[-1]
    if t <= max_frames:
        return mel
    if torch.is_grad_enabled():
        start = torch.randint(0, t - max_frames + 1, (1,)).item()
    else:
        start = max(0, (t - max_frames) // 2)
    return mel[:, :, start : start + max_frames]
