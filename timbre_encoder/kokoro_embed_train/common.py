"""Shared helpers for Kokoro 2-step embedding training."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

from timbre_encoder.aligner import FrozenTextAligner  # noqa: E402
from timbre_encoder.voice_bank import CORE_VOICES  # noqa: E402

MAX_WAV_SEC = 8.0
DEFAULT_BASELINE = "af_heart"

__all__ = [
    "CORE_VOICES",
    "DEFAULT_BASELINE",
    "MAX_WAV_SEC",
    "calibrate_f0_voiced",
    "calibrate_st2_to_kokoro",
    "corr",
    "freeze_module",
    "match_mean_std",
]


def freeze_module(m: torch.nn.Module) -> None:
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)


def resize_1d(x: torch.Tensor, t: int) -> torch.Tensor:
    x = x.float().view(1, 1, -1)
    return F.interpolate(x, size=t, mode="linear", align_corners=False).view(-1)


def match_mean_std(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    s = src.float().view(-1)
    r = ref.float().view(-1).detach()
    if s.numel() != r.numel():
        s = resize_1d(s, r.numel())
    return (s - s.mean()) / s.std().clamp_min(1e-5) * r.std().clamp_min(1e-5) + r.mean()


def calibrate_f0_voiced(
    f0_src: torch.Tensor,
    f0_kokoro: torch.Tensor,
    src_voiced: float = 1.0,
    kokoro_voiced: float = 50.0,
) -> torch.Tensor:
    y = resize_1d(f0_src.float(), f0_kokoro.numel())
    k = f0_kokoro.float().view(-1)
    y_v = y > src_voiced
    k_v = k > kokoro_voiced
    if int(y_v.sum()) >= 4 and int(k_v.sum()) >= 4:
        ys, ks = y[y_v], k[k_v]
        out = (y - ys.mean()) / ys.std().clamp_min(1e-5) * ks.std().clamp_min(1e-5) + ks.mean()
    else:
        out = match_mean_std(y, k)
    out = out.clone()
    out[~y_v] = 0.0
    return out


def corr(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    x, y = a.float().view(-1), b.float().view(-1)
    if mask is not None:
        m = mask.bool().view(-1)
        if int(m.sum()) < 4:
            return float("nan")
        x, y = x[m], y[m]
    if x.numel() != y.numel():
        n = min(x.numel(), y.numel())
        x, y = x[:n], y[:n]
    x, y = x - x.mean(), y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom) < 1e-8:
        return float("nan")
    return float((x * y).sum() / denom)


def mae(a: torch.Tensor, b: torch.Tensor) -> float:
    x, y = a.float().view(-1), b.float().view(-1)
    n = min(x.numel(), y.numel())
    return float((x[:n] - y[:n]).abs().mean())


def kokoro_ids(model, phonemes: str, device: torch.device):
    dropped: list[str] = []
    ids: list[int] = []
    for p in phonemes:
        i = model.vocab.get(p)
        if i is None:
            dropped.append(p)
        else:
            ids.append(i)
    input_ids = torch.LongTensor([[0, *ids, 0]]).to(device)
    return input_ids, ids, dropped


def build_alignment_from_d_gt(
    d_gt: Sequence[int] | torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """BOS/EOS dur=1 + phoneme d_gt → [1, N+2, T]."""
    if isinstance(d_gt, torch.Tensor):
        inner = [max(1, int(round(float(x)))) for x in d_gt.tolist()]
    else:
        inner = [max(1, int(x)) for x in d_gt]
    d_list = [1] + inner + [1]
    return FrozenTextAligner.d_gt_to_a_gt(d_list, device=device)


def pack_slot(baseline_pack: torch.Tensor, phoneme_len: int) -> torch.Tensor:
    """Official pack [510,1,256] → ref_s [1, 256] by phoneme length."""
    idx = max(0, min(phoneme_len - 1, baseline_pack.shape[0] - 1))
    slot = baseline_pack[idx]
    if slot.dim() == 1:
        slot = slot.unsqueeze(0)
    elif slot.dim() == 2 and slot.shape[0] == 1:
        pass
    else:
        slot = slot.view(1, -1)
    return slot


def prepare_utt(
    model,
    phonemes_st2: str,
    d_gt: Sequence[int] | torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int] | None:
    """
    Returns (input_ids, alignment, n_phonemes) or None if vocab mismatch.
    n_phonemes = len(phonemes_st2) without BOS/EOS.
    """
    input_ids, ids, dropped = kokoro_ids(model, phonemes_st2, device)
    if dropped or len(ids) != len(phonemes_st2):
        return None
    if isinstance(d_gt, torch.Tensor):
        d_list = [int(round(float(x))) for x in d_gt.tolist()]
    else:
        d_list = [int(x) for x in d_gt]
    if len(d_list) != len(phonemes_st2):
        return None
    aln = build_alignment_from_d_gt(d_list, device)
    if aln.shape[1] != input_ids.shape[1]:
        return None
    return input_ids, aln, len(phonemes_st2)


@torch.no_grad()
def prosody_heads_forward(
    model,
    input_ids: torch.Tensor,
    ref_s: torch.Tensor,
    alignment: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Duration soft prediction + F0/N with GT alignment. No decoder.
    Returns pred_dur [N], F0 [1,T] or [T], N [1,T] or [T].
    """
    device = input_ids.device
    input_lengths = torch.full(
        (input_ids.shape[0],),
        input_ids.shape[-1],
        device=device,
        dtype=torch.long,
    )
    text_mask = torch.arange(input_lengths.max(), device=device).unsqueeze(0).expand(
        input_lengths.shape[0], -1
    )
    text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))
    bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
    s = ref_s[:, 128:]
    d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
    x, _ = model.predictor.lstm(d)
    duration = model.predictor.duration_proj(x)
    pred_dur = torch.sigmoid(duration).sum(axis=-1).squeeze(0)  # [N]

    pred_aln = alignment.to(device)
    if pred_aln.dim() == 2:
        pred_aln = pred_aln.unsqueeze(0)
    en = d.transpose(-1, -2) @ pred_aln
    f0, n = model.predictor.F0Ntrain(en, s)
    return pred_dur, f0, n


def prosody_heads_forward_grad(
    model,
    input_ids: torch.Tensor,
    ref_s: torch.Tensor,
    alignment: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same as prosody_heads_forward but allows grads into ref_s."""
    device = input_ids.device
    input_lengths = torch.full(
        (input_ids.shape[0],),
        input_ids.shape[-1],
        device=device,
        dtype=torch.long,
    )
    text_mask = torch.arange(input_lengths.max(), device=device).unsqueeze(0).expand(
        input_lengths.shape[0], -1
    )
    text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))
    bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
    s = ref_s[:, 128:]
    d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
    x, _ = model.predictor.lstm(d)
    duration = model.predictor.duration_proj(x)
    pred_dur = torch.sigmoid(duration).sum(axis=-1).squeeze(0)

    pred_aln = alignment.to(device)
    if pred_aln.dim() == 2:
        pred_aln = pred_aln.unsqueeze(0)
    en = d.transpose(-1, -2) @ pred_aln
    f0, n = model.predictor.F0Ntrain(en, s)
    return pred_dur, f0, n


def to_1d(x: torch.Tensor) -> torch.Tensor:
    x = x.float().squeeze()
    if x.dim() > 1:
        x = x[0]
    return x.view(-1)


@torch.no_grad()
def kokoro_f0n_for_ref(
    model,
    input_ids: torch.Tensor,
    ref_s: torch.Tensor,
    alignment: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, f0, n = prosody_heads_forward(model, input_ids, ref_s, alignment)
    return to_1d(f0), to_1d(n)


def calibrate_st2_to_kokoro(
    f0_st2: torch.Tensor,
    n_st2: torch.Tensor,
    f0_ref: torch.Tensor,
    n_ref: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map ST2 cache curves into Kokoro units using a Kokoro reference curve."""
    device = f0_ref.device
    f0_cal = calibrate_f0_voiced(f0_st2.to(device), f0_ref)
    n_cal = match_mean_std(n_st2.to(device), n_ref)
    return f0_cal, n_cal


@torch.no_grad()
def multi_voice_kokoro_mean_f0n(
    model,
    input_ids: torch.Tensor,
    alignment: torch.Tensor,
    packs: dict[str, torch.Tensor],
    phoneme_len: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """
    Run F0Ntrain per voice pack; return mean F0/N curves + per-voice dict.
    packs: name -> [510,1,256]
    """
    per: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    f0s: list[torch.Tensor] = []
    ns: list[torch.Tensor] = []
    for name, pack in packs.items():
        ref = pack_slot(pack, phoneme_len).to(input_ids.device)
        f0, n = kokoro_f0n_for_ref(model, input_ids, ref, alignment)
        per[name] = (f0, n)
        f0s.append(f0)
        ns.append(n)
    # Align lengths (should match); stack mean
    t = min(x.numel() for x in f0s)
    f0_mean = torch.stack([x[:t] for x in f0s], dim=0).mean(dim=0)
    n_mean = torch.stack([x[:t] for x in ns], dim=0).mean(dim=0)
    return f0_mean, n_mean, per


def load_voice_packs(pipeline, names: Iterable[str], device: torch.device) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name in names:
        pack = pipeline.load_voice(name).to(device)
        out[name] = pack
    return out


def save_ckpt(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_ckpt(path: str | Path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def d_gt_target_with_bos_eos(d_gt: Sequence[int] | torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(d_gt, torch.Tensor):
        inner = [max(1, int(round(float(x)))) for x in d_gt.tolist()]
    else:
        inner = [max(1, int(x)) for x in d_gt]
    return torch.tensor([1] + inner + [1], device=device, dtype=torch.float32)
