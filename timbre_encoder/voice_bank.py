"""Statistics over official Kokoro voice packs (mean / std bank)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from kokoro.pipeline import KPipeline

# American English presets (Kokoro-82M VOICES.md)
US_VOICES: list[str] = [
    "af_heart",
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
]

# Smaller fast bank (high-grade voices)
CORE_VOICES: list[str] = [
    "af_heart",
    "af_bella",
    "af_sarah",
    "af_nicole",
    "am_michael",
    "am_puck",
    "am_fenrir",
    "bf_emma",
    "bm_george",
]


@dataclass
class VoiceBank:
    """Mean pack + per-dim mean/std for timbre and prosody halves."""

    names: list[str]
    mean_pack: torch.Tensor  # [510, 1, 256]
    timbre_mean: torch.Tensor  # [128]
    timbre_std: torch.Tensor  # [128]
    prosody_mean: torch.Tensor  # [128]
    prosody_std: torch.Tensor  # [128]

    def to(self, device: torch.device | str) -> "VoiceBank":
        return VoiceBank(
            names=self.names,
            mean_pack=self.mean_pack.to(device),
            timbre_mean=self.timbre_mean.to(device),
            timbre_std=self.timbre_std.to(device),
            prosody_mean=self.prosody_mean.to(device),
            prosody_std=self.prosody_std.to(device),
        )


def load_voice_packs(
    pipeline: KPipeline,
    names: Sequence[str],
) -> tuple[list[str], list[torch.Tensor]]:
    loaded: list[str] = []
    packs: list[torch.Tensor] = []
    for name in names:
        try:
            pack = pipeline.load_voice(name)
            if pack.dim() == 2:
                pack = pack.unsqueeze(1)
            packs.append(pack.detach().float().cpu())
            loaded.append(name)
            print(f"  voice_bank + {name} {tuple(packs[-1].shape)}", flush=True)
        except Exception as e:
            print(f"  voice_bank skip {name}: {type(e).__name__}: {e}", flush=True)
    if not packs:
        raise RuntimeError("No voices loaded for bank")
    return loaded, packs


def build_voice_bank(
    pipeline: KPipeline,
    names: Sequence[str] | None = None,
) -> VoiceBank:
    """
    Stack all packs → mean pack [510,1,256].
    Timbre/prosody mean&std over all voices × all slots (per dimension).
    """
    names = list(names) if names is not None else list(US_VOICES)
    print(f"Building voice bank from {len(names)} names...", flush=True)
    loaded, packs = load_voice_packs(pipeline, names)
    stacked = torch.stack(packs, dim=0)  # [V, 510, 1, 256]
    mean_pack = stacked.mean(dim=0)  # [510, 1, 256]

    timbre = stacked[:, :, 0, :128].reshape(-1, 128)  # [V*510, 128]
    prosody = stacked[:, :, 0, 128:].reshape(-1, 128)
    timbre_mean = timbre.mean(dim=0)
    timbre_std = timbre.std(dim=0).clamp_min(1e-4)
    prosody_mean = prosody.mean(dim=0)
    prosody_std = prosody.std(dim=0).clamp_min(1e-4)

    print(
        f"voice_bank n={len(loaded)} timbre_std_mean={float(timbre_std.mean()):.4f} "
        f"prosody_std_mean={float(prosody_std.mean()):.4f}",
        flush=True,
    )
    return VoiceBank(
        names=loaded,
        mean_pack=mean_pack,
        timbre_mean=timbre_mean,
        timbre_std=timbre_std,
        prosody_mean=prosody_mean,
        prosody_std=prosody_std,
    )


@torch.no_grad()
def normalize_to_bank(
    vec: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    z_clip: float = 2.5,
) -> torch.Tensor:
    """
    Project vector onto bank distribution: per-dim z-score, clip, de-whiten.

    vec can be [128] or any shape ending in 128 (last dim normalized).
    """
    mean = mean.to(device=vec.device, dtype=vec.dtype)
    std = std.to(device=vec.device, dtype=vec.dtype).clamp_min(1e-4)
    flat = vec.reshape(-1, mean.numel())
    z = (flat - mean) / std
    if z_clip > 0:
        z = z.clamp(-z_clip, z_clip)
    out = z * std + mean
    return out.reshape_as(vec)


def save_bank(bank: VoiceBank, path: str | torch.serialization.PATH) -> None:
    torch.save(
        {
            "names": bank.names,
            "mean_pack": bank.mean_pack.cpu(),
            "timbre_mean": bank.timbre_mean.cpu(),
            "timbre_std": bank.timbre_std.cpu(),
            "prosody_mean": bank.prosody_mean.cpu(),
            "prosody_std": bank.prosody_std.cpu(),
        },
        path,
    )


def load_bank(path: str | torch.serialization.PATH, device=None) -> VoiceBank:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    bank = VoiceBank(
        names=list(obj["names"]),
        mean_pack=obj["mean_pack"],
        timbre_mean=obj["timbre_mean"],
        timbre_std=obj["timbre_std"],
        prosody_mean=obj["prosody_mean"],
        prosody_std=obj["prosody_std"],
    )
    if device is not None:
        bank = bank.to(device)
    return bank
