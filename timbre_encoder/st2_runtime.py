"""Frozen StyleTTS2-LibriTTS stack helpers for clean-path encoder training."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from munch import Munch

ROOT = Path(__file__).resolve().parents[1]
ST2_ROOT = ROOT / "third_party" / "StyleTTS2"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints" / "styletts2_libritts" / "Models" / "LibriTTS"
_THIRD = ROOT / "third_party"


def _ensure_st2_path() -> None:
    for p in (str(_THIRD), str(ST2_ROOT), str(ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)


def recursive_munch(d: Any) -> Any:
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    if isinstance(d, list):
        return [recursive_munch(v) for v in d]
    return d


def length_to_mask(lengths: torch.Tensor) -> torch.Tensor:
    """StyleTTS2 padding mask: True = pad."""
    mask = torch.arange(lengths.max(), device=lengths.device).unsqueeze(0).expand(lengths.shape[0], -1)
    return torch.gt(mask + 1, lengths.unsqueeze(1))


def patch_torch_load() -> None:
    """ST2 checkpoints need weights_only=False on PyTorch >= 2.6."""
    if getattr(torch.load, "_st2_patched", False):
        return
    _orig = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)

    _load._st2_patched = True  # type: ignore[attr-defined]
    torch.load = _load  # type: ignore[assignment]


def load_frozen_st2(device: torch.device):
    """Load StyleTTS2-LibriTTS; all params frozen, eval mode."""
    _ensure_st2_path()
    patch_torch_load()

    from models import build_model, load_ASR_models, load_F0_models
    from Utils.PLBERT.util import load_plbert

    config = yaml.safe_load(open(CKPT_DIR / "config.yml", encoding="utf-8"))
    text_aligner = load_ASR_models(str(ST2_ROOT / config["ASR_path"]), str(ST2_ROOT / config["ASR_config"]))
    pitch_extractor = load_F0_models(str(ST2_ROOT / config["F0_path"]))
    plbert = load_plbert(str(ST2_ROOT / config["PLBERT_dir"]))

    model = build_model(recursive_munch(config["model_params"]), text_aligner, pitch_extractor, plbert)
    params = torch.load(CKPT_DIR / "epochs_2nd_00020.pth", map_location="cpu", weights_only=False)
    params = params["net"] if "net" in params else params
    for key in model:
        if key not in params:
            continue
        try:
            model[key].load_state_dict(params[key])
        except Exception:
            state = {k[7:] if k.startswith("module.") else k: v for k, v in params[key].items()}
            model[key].load_state_dict(state, strict=False)
        model[key].eval().to(device)
        for p in model[key].parameters():
            p.requires_grad = False
    return model, config


def phonemes_to_st2_ids(phonemes: str, device: torch.device) -> torch.Tensor:
    from timbre_encoder.aligner import TextCleaner

    ids = TextCleaner()(phonemes)
    if not ids:
        ids = [0]
    return torch.LongTensor(ids).unsqueeze(0).to(device)


def prepare_asr_f0_n(
    model,
    mel: torch.Tensor,
    text_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Frozen teacher-forced features for decoder.
    Returns (asr, F0_full, N_full, mel_even) — all detached except mel_even reference.
    """
    if mel.dim() == 2:
        mel = mel.unsqueeze(0)
    if mel.shape[-1] % 2 == 1:
        mel = mel[..., :-1]

    device = mel.device
    texts = text_ids
    input_lengths = torch.LongTensor([texts.shape[1]]).to(device)
    mel_input_length = torch.LongTensor([mel.shape[-1]]).to(device)
    text_mask = length_to_mask(input_lengths)
    n_down = model.text_aligner.n_down
    mel_down = torch.clamp(mel_input_length // (2**n_down), min=1)
    mel_mask = length_to_mask(mel_down)

    with torch.no_grad():
        _, _, s2s_attn = model.text_aligner(mel, mel_mask, texts)
        s2s_attn = s2s_attn.transpose(-1, -2)[..., 1:].transpose(-1, -2)
        from monotonic_align import mask_from_lens, maximum_path

        mask_st = mask_from_lens(s2s_attn, input_lengths, mel_down)
        s2s_attn_mono = maximum_path(s2s_attn, mask_st)
        t_en = model.text_encoder(texts, input_lengths, text_mask)
        asr = t_en @ s2s_attn_mono

        F0_real, _, _ = model.pitch_extractor(mel.unsqueeze(1))
        from utils import log_norm

        real_norm = log_norm(mel.unsqueeze(1)).squeeze(1)
        if F0_real.dim() == 1:
            F0_real = F0_real.unsqueeze(0)
        if real_norm.dim() == 1:
            real_norm = real_norm.unsqueeze(0)

        T_asr = asr.shape[-1]
        T_f0 = 2 * T_asr

        def _match_full(x: torch.Tensor) -> torch.Tensor:
            if x.shape[-1] == T_f0:
                return x
            return F.interpolate(x.unsqueeze(1), size=T_f0, mode="linear", align_corners=False).squeeze(1)

        F0_use = _match_full(F0_real)
        N_use = _match_full(real_norm)

    return asr.detach(), F0_use.detach(), N_use.detach(), mel


def decode_with_style(model, asr: torch.Tensor, F0: torch.Tensor, N: torch.Tensor, s_a: torch.Tensor) -> torch.Tensor:
    """
    Decoder forward; s_a may require grad. Frozen decoder weights.
    Returns waveform [B, T] or [T].
    """
    return model.decoder(asr, F0, N, s_a)
