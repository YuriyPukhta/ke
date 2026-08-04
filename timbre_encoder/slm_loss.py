"""Frozen WavLM feature-matching loss (StyleTTS2 SLM path; WavLM not tuned).

Designed to run on CPU while Kokoro stays on CUDA: pass device='cpu'.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class WavLMFeatureLoss(nn.Module):
    """
    L_slm ≈ mean L1 on selected WavLM hidden layers.
    Optional: + (1 - cosine) on mean-pooled features per layer.
    WavLM weights stay frozen. Gradients flow into pred waveform.
    """

    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base-plus",
        layers: Optional[list[int]] = None,
        device: str | torch.device = "cpu",
        include_cos: bool = False,
    ):
        super().__init__()
        from transformers import WavLMModel

        self._device = torch.device(device)
        self.wavlm = WavLMModel.from_pretrained(model_name)
        self.wavlm.eval()
        for p in self.wavlm.parameters():
            p.requires_grad = False
        self.layers = layers or [4, 8, 12]
        self.include_cos = include_cos
        self.sample_rate = 16000
        self.to(self._device)

    def _resample_24k_to_16k(self, wav: torch.Tensor) -> torch.Tensor:
        while wav.dim() > 2:
            wav = wav.squeeze(1) if wav.shape[1] == 1 else wav.reshape(wav.shape[0], -1)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        n = int(wav.shape[-1] * 16000 / 24000)
        return F.interpolate(wav.unsqueeze(1), size=n, mode="linear", align_corners=False).squeeze(1)

    def _match_length(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = min(a.shape[-1], b.shape[-1])
        return a[..., :n], b[..., :n]

    def forward(self, pred_wav_24k: torch.Tensor, real_wav_24k: torch.Tensor) -> torch.Tensor:
        # Move waves to WavLM device (CPU); .to() keeps autograd across devices.
        pred = self._resample_24k_to_16k(pred_wav_24k.to(self._device))
        real = self._resample_24k_to_16k(real_wav_24k.to(self._device))
        pred, real = self._match_length(pred, real)

        with torch.no_grad():
            real_hs = self.wavlm(real, output_hidden_states=True).hidden_states

        # Grad through WavLM into pred waveform (params frozen).
        pred_hs = self.wavlm(pred, output_hidden_states=True).hidden_states

        l1_losses: list[torch.Tensor] = []
        cos_losses: list[torch.Tensor] = []
        for li in self.layers:
            if li >= len(pred_hs):
                continue
            ph = pred_hs[li]
            rh = real_hs[li].detach()
            l1_losses.append(F.l1_loss(ph, rh))
            if self.include_cos:
                # [B, T, C] → mean over time → [B, C]
                fa = ph.mean(dim=1)
                fb = rh.mean(dim=1)
                cos = F.cosine_similarity(fa, fb, dim=-1).mean()
                cos_losses.append(1.0 - cos)
        if not l1_losses:
            return pred_wav_24k.new_zeros(())
        loss = torch.stack(l1_losses).mean()
        if cos_losses:
            loss = loss + torch.stack(cos_losses).mean()
        return loss

    @torch.no_grad()
    def similarity(self, wav_a_24k: torch.Tensor, wav_b_24k: torch.Tensor) -> dict[str, float]:
        """
        Voice similarity vs reference: cosine of mean-pooled WavLM features
        (higher better) and feature L1 (lower better).
        """
        a = self._resample_24k_to_16k(wav_a_24k.to(self._device))
        b = self._resample_24k_to_16k(wav_b_24k.to(self._device))
        a, b = self._match_length(a, b)
        ha = self.wavlm(a, output_hidden_states=True).hidden_states
        hb = self.wavlm(b, output_hidden_states=True).hidden_states
        cos_vals: list[float] = []
        l1_vals: list[float] = []
        for li in self.layers:
            if li >= len(ha):
                continue
            # [B, T, C] → mean over time → [B, C]
            fa = ha[li].mean(dim=1)
            fb = hb[li].mean(dim=1)
            cos_vals.append(float(F.cosine_similarity(fa, fb, dim=-1).mean()))
            l1_vals.append(float(F.l1_loss(fa, fb)))
        return {
            "wavlm_cos": sum(cos_vals) / max(len(cos_vals), 1),
            "wavlm_l1": sum(l1_vals) / max(len(l1_vals), 1),
        }
