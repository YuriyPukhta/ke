"""Assemble Kokoro voice packs from timbre + baseline prosody."""
from __future__ import annotations

import torch


def normalize_timbre(timbre: torch.Tensor) -> torch.Tensor:
    """Flatten to [128]."""
    t = timbre.reshape(-1)
    if t.numel() != 128:
        raise ValueError(f"Expected 128 timbre dims, got {t.numel()}")
    return t


def assemble_voice_pack(
    timbre: torch.Tensor,
    baseline_pack: torch.Tensor,
    num_slots: int = 510,
) -> torch.Tensor:
    """
    Build [510, 1, 256] voice pack for KPipeline.

    timbre: optimized or encoded timbre, any shape ending in 128
    baseline_pack: official voice e.g. af_heart [510, 1, 256]
    """
    if baseline_pack.shape[0] < num_slots:
        raise ValueError(f"baseline_pack needs at least {num_slots} slots, got {baseline_pack.shape[0]}")
    t = normalize_timbre(timbre)
    pack = baseline_pack[:num_slots].clone()
    pack[:, :, :128] = t.view(1, 1, 128)
    return pack


def make_ref_s(
    timbre: torch.Tensor,
    baseline_pack: torch.Tensor,
    phoneme_length: int,
) -> torch.Tensor:
    """Single inference ref_s [1, 256] for phoneme string length."""
    pack = assemble_voice_pack(timbre, baseline_pack)
    idx = max(0, min(phoneme_length - 1, pack.shape[0] - 1))
    return pack[idx]
