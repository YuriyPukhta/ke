"""Shared helpers for TMA ← Kokoro duration distillation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

from gt_cache.jdc_finetune.common import (  # noqa: E402
    ALL_KOKORO_VOICES,
    DEFAULT_TEXTS,
    DEFAULT_TRAIN_VOICES,
    DEFAULT_VAL_VOICES,
    list_kokoro_voices_from_cache,
    phonemes_from_text,
    pick_val_voices,
    tokens_from_phonemes,
    write_jsonl,
    read_jsonl,
)
from timbre_encoder.aligner import (  # noqa: E402
    ASRCNN,
    TextCleaner,
    ST2_SYMBOL_TO_ID,
    mask_from_lens,
    maximum_path_torch,
)
from timbre_encoder.phoneme_norm import normalize_kokoro_phonemes_for_st2  # noqa: E402

DEFAULT_ASR_CKPT = (
    ROOT / "timbre_encoder" / "checkpoints" / "asr_epoch_00080.pth"
)
DEFAULT_ASR_CFG = ROOT / "third_party" / "styletts2_asr" / "config.yml"


def phonemes_st2_from_text(pipeline, text: str) -> str | None:
    raw = phonemes_from_text(pipeline, text)
    if not raw:
        return None
    return normalize_kokoro_phonemes_for_st2(raw).normalized


def text_ids_from_phonemes(phonemes: str) -> list[int]:
    cleaner = TextCleaner()
    ids = cleaner(phonemes)
    return ids if ids else [0]


def rescale_durs_to_t(durs: list[int] | np.ndarray, t_target: int) -> list[int]:
    """Largest-remainder rescale so durations sum exactly to t_target (min 1 each if possible)."""
    d = np.asarray([max(0.0, float(x)) for x in durs], dtype=np.float64)
    n = len(d)
    if n == 0:
        return []
    t_target = max(int(t_target), n)  # at least 1 frame per phone when possible
    if d.sum() < 1e-8:
        d = np.ones(n, dtype=np.float64)
    scaled = d / d.sum() * t_target
    base = np.floor(scaled).astype(np.int64)
    # ensure every phone gets >= 1 when t_target >= n
    for i in range(n):
        if base[i] < 1:
            base[i] = 1
    # fix sum
    diff = int(t_target - int(base.sum()))
    if diff != 0:
        frac = scaled - np.floor(scaled)
        order = np.argsort(-frac) if diff > 0 else np.argsort(frac)
        step = 1 if diff > 0 else -1
        i = 0
        guard = 0
        while diff != 0 and guard < n * 4:
            idx = int(order[i % n])
            if step < 0 and base[idx] <= 1:
                i += 1
                guard += 1
                continue
            base[idx] += step
            diff -= step
            i += 1
            guard += 1
    # final safety
    if int(base.sum()) != t_target:
        base[-1] += t_target - int(base.sum())
        if base[-1] < 1:
            base[-1] = 1
    return [int(x) for x in base.tolist()]


def durs_to_hard(durs: list[int], device=None) -> torch.Tensor:
    """[N] → hard a_gt [1, N, T]."""
    d_list = [max(0, int(x)) for x in durs]
    t = max(1, sum(d_list))
    a = torch.zeros(1, len(d_list), t, device=device)
    cur = 0
    for i, dur in enumerate(d_list):
        if dur > 0:
            a[0, i, cur : cur + dur] = 1.0
            cur += dur
    return a


def soft_attn_nll(soft: torch.Tensor, hard: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    soft/hard: [B, N, T] (or [N,T]). Hard is 0/1 MAS path.
    Encourages soft mass on hard frames.
    """
    if soft.dim() == 2:
        soft = soft.unsqueeze(0)
        hard = hard.unsqueeze(0)
    t = min(soft.shape[-1], hard.shape[-1])
    n = min(soft.shape[1], hard.shape[1])
    soft = soft[:, :n, :t]
    hard = hard[:, :n, :t]
    denom = hard.sum().clamp_min(1.0)
    return -(hard * (soft.clamp_min(eps).log())).sum() / denom


def duration_corr(a: list[int] | torch.Tensor, b: list[int] | torch.Tensor) -> float:
    if isinstance(a, torch.Tensor):
        a = [float(x) for x in a.detach().cpu().view(-1).tolist()]
    if isinstance(b, torch.Tensor):
        b = [float(x) for x in b.detach().cpu().view(-1).tolist()]
    n = min(len(a), len(b))
    if n < 2:
        return float("nan")
    x = np.asarray(a[:n], dtype=np.float64)
    y = np.asarray(b[:n], dtype=np.float64)
    if x.std() < 1e-9 or y.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def load_asr_trainable(
    device: torch.device,
    weights: Path | None = None,
    config_path: Path | None = None,
) -> ASRCNN:
    cfg_path = Path(config_path or DEFAULT_ASR_CFG)
    ckpt = Path(weights or DEFAULT_ASR_CKPT)
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    mp = cfg["model_params"]
    model = ASRCNN(
        input_dim=mp["input_dim"],
        hidden_dim=mp["hidden_dim"],
        n_token=mp["n_token"],
        token_embedding_dim=mp["token_embedding_dim"],
    )
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model_state = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(model_state, strict=False)
    # disable token dropout for distillation
    if hasattr(model, "asr_s2s"):
        model.asr_s2s.random_mask = 0.0
    model.to(device)
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    print(f"loading ASRCNN from {ckpt}", flush=True)
    return model


def save_asr_checkpoint(model: ASRCNN, path: Path, meta: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
    if meta:
        payload["meta"] = meta
    torch.save(payload, path)
    print(f"saved {path}", flush=True)


def forward_soft_attn(
    model: ASRCNN,
    mel: torch.Tensor,
    text_ids: torch.Tensor,
) -> torch.Tensor:
    """
    mel: [1, 80, T] or [80, T]
    text_ids: [N] or [1, N]
    returns soft attn [1, N, T_down] (BOS dropped)
    """
    if mel.dim() == 2:
        mel = mel.unsqueeze(0)
    if mel.shape[-1] % 2 == 1:
        mel = mel[..., :-1]
    if text_ids.dim() == 1:
        text_ids = text_ids.unsqueeze(0)
    device = mel.device
    n_down = int(model.n_down)
    mel_len = torch.tensor([mel.shape[-1]], device=device)
    mel_len_down = torch.clamp(mel_len // (2**n_down), min=1)
    mask = model.length_to_mask(mel_len_down)
    _, _, alignments = model(mel, mask, text_ids)
    attn = alignments[:, 1 : 1 + text_ids.shape[1], : int(mel_len_down.item())].contiguous()
    return attn


@torch.no_grad()
def soft_to_durs(soft: torch.Tensor) -> list[int]:
    """MAS on soft attn → per-phoneme durations."""
    if soft.dim() == 2:
        soft = soft.unsqueeze(0)
    text_len = torch.tensor([soft.shape[1]], device=soft.device)
    mel_len = torch.tensor([soft.shape[2]], device=soft.device)
    try:
        from monotonic_align import mask_from_lens as mas_mask
        from monotonic_align import maximum_path as mas_path

        mask_st = mas_mask(soft, text_len, mel_len)
        a_gt = mas_path(soft, mask_st)
    except Exception:
        mask_st = mask_from_lens(soft, text_len, mel_len)
        a_gt = maximum_path_torch(soft * mask_st.float(), mask_st.float())
    d = a_gt.sum(dim=-1).squeeze(0)
    return [max(1, int(round(float(x)))) for x in d.tolist()]
