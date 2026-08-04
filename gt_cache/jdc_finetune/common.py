"""Shared helpers for JDC ← Kokoro F0 distillation."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

from gt_cache.utils import CKPT_DIR, ST2_ROOT, load_pitch_extractor  # noqa: E402
from timbre_encoder.st2_runtime import patch_torch_load  # noqa: E402

# All English Kokoro-82M packs (HF voices/*.pt). Updated if cache grows via list_kokoro_voices().
ALL_KOKORO_VOICES = (
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_heart",
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
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
)

# Held-out validation voices (10): never in train. Balanced af/am/bf/bm.
DEFAULT_VAL_VOICES = (
    "af_nicole",
    "af_jessica",
    "af_nova",
    "am_adam",
    "am_echo",
    "am_onyx",
    "bf_alice",
    "bf_lily",
    "bm_fable",
    "bm_lewis",
)

DEFAULT_TRAIN_VOICES = tuple(v for v in ALL_KOKORO_VOICES if v not in set(DEFAULT_VAL_VOICES))

# Back-compat alias (train + val).
DEFAULT_VOICES = DEFAULT_TRAIN_VOICES + DEFAULT_VAL_VOICES


def list_kokoro_voices_from_cache() -> list[str]:
    """Discover voice stems from local HF Kokoro-82M cache (fallback: ALL_KOKORO_VOICES)."""
    hub = Path.home() / ".cache" / "huggingface" / "hub" / "models--hexgrad--Kokoro-82M" / "snapshots"
    found: set[str] = set()
    if hub.is_dir():
        for snap in hub.iterdir():
            voices = snap / "voices"
            if voices.is_dir():
                for p in voices.glob("*.pt"):
                    found.add(p.stem)
    if found:
        return sorted(found)
    return list(ALL_KOKORO_VOICES)


def pick_val_voices(all_voices: list[str], n_val: int, seed: int = 0) -> tuple[list[str], list[str]]:
    """
    Pick n_val held-out voices, preferring DEFAULT_VAL_VOICES order, then fill by locale buckets.
    Returns (train_voices, val_voices).
    """
    all_set = list(dict.fromkeys(all_voices))  # stable unique
    if n_val <= 0:
        return all_set, []
    if n_val >= len(all_set):
        raise ValueError(f"n_val={n_val} >= n_voices={len(all_set)}")

    preferred = [v for v in DEFAULT_VAL_VOICES if v in all_set]
    val: list[str] = []
    for v in preferred:
        if len(val) >= n_val:
            break
        val.append(v)

    if len(val) < n_val:
        # fill remaining by locale prefix round-robin for diversity
        buckets = {"af": [], "am": [], "bf": [], "bm": [], "other": []}
        for v in all_set:
            if v in val:
                continue
            key = v[:2] if v[:2] in buckets else "other"
            buckets[key].append(v)
        rng = random.Random(seed)
        for b in buckets.values():
            rng.shuffle(b)
        keys = [k for k, b in buckets.items() if b]
        i = 0
        while len(val) < n_val and keys:
            k = keys[i % len(keys)]
            if buckets[k]:
                val.append(buckets[k].pop())
            else:
                keys = [x for x in keys if buckets[x]]
                continue
            i += 1

    val_set = set(val)
    train = [v for v in all_set if v not in val_set]
    return train, val


DEFAULT_TEXTS = [
    "Hello, this is a short test of pitch and energy.",
    "The quick brown fox jumps over the lazy dog near the river.",
    "I would like to know whether the extracted contours match Kokoro.",
    "Please call Stella. Ask her to bring these things with her from the store.",
    "She can scoop these things into three red bags, and we will go meet her Wednesday at the train station.",
    "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow.",
    "The rainbow is a division of white light into many beautiful colors.",
    "These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon.",
    "There is, according to legend, a boiling pot of gold at one end.",
    "People look, but no one ever finds it.",
    "When a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow.",
    "Throughout the centuries people have explained the rainbow in various ways.",
    "Some have accepted it as a miracle without physical explanation.",
    "To the Hebrews it was a token that there would be no more universal floods.",
    "The Greeks used to imagine that it was a sign from the gods to foretell war or heavy rain.",
    "The Norsemen considered the rainbow as a bridge over which the gods passed from earth to their home in the sky.",
    "Others have tried to explain the phenomenon physically.",
    "Aristotle thought that the rainbow was caused by reflection of the sun's rays by the rain.",
    "Since then physicists have found that it is not reflection, but refraction by the raindrops which causes the rainbows.",
    "Many complicated ideas about the rainbow have been formed.",
    "The difference in the rainbow depends considerably upon the size of the drops, and the width of the colored band increases as the size of the drops increases.",
    "The actual primary rainbow observed is said to be the effect of superimposition of a number of bows.",
    "If the red of the second bow falls upon the green of the first, the result is to give a bow with an abnormally wide yellow band.",
    "Under favorable conditions a secondary bow may be seen, which is much fainter and has its colors reversed.",
    "This secondary rainbow is formed by rays that have suffered two internal reflections.",
]


def resize_1d(x: torch.Tensor, t: int) -> torch.Tensor:
    x = x.float().view(1, 1, -1)
    return F.interpolate(x, size=t, mode="linear", align_corners=False).view(-1)


def pred_dur_to_alignment(pred_dur: torch.Tensor, device: torch.device) -> torch.Tensor:
    d = pred_dur.detach().float().view(-1).clamp_min(1)
    d_i = torch.round(d).long().clamp_min(1)
    n = int(d_i.numel())
    t = int(d_i.sum().item())
    aln = torch.zeros(1, n, t, device=device)
    col = 0
    for i in range(n):
        span = int(d_i[i].item())
        aln[0, i, col : col + span] = 1.0
        col += span
    return aln


def phonemes_from_text(pipeline, text: str) -> str | None:
    for _, ps, _ in pipeline.en_tokenize(pipeline.g2p(text)[1]):
        return ps
    return None


def tokens_from_phonemes(model, phonemes: str) -> list[int] | None:
    ids = [j for j in (model.vocab.get(p) for p in phonemes) if j is not None]
    if len(ids) != len(phonemes):
        return None
    if len(ids) + 2 > model.context_length:
        return None
    return ids


def corr_voiced(
    pred: torch.Tensor,
    target: torch.Tensor,
    voiced_thresh: float = 50.0,
) -> float:
    x = pred.float().view(-1)
    y = target.float().view(-1)
    n = min(x.numel(), y.numel())
    x, y = x[:n], y[:n]
    m = y > voiced_thresh
    if int(m.sum()) < 4:
        return float("nan")
    x, y = x[m], y[m]
    x, y = x - x.mean(), y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom) < 1e-8:
        return float("nan")
    return float((x * y).sum() / denom)


def load_jdc_trainable(device: torch.device, weights: Path | None = None):
    """Load JDCNet with gradients enabled (for fine-tune)."""
    patch_torch_load()
    from models import load_F0_models
    import yaml

    config = yaml.safe_load(open(CKPT_DIR / "config.yml", encoding="utf-8"))
    f0_path = Path(weights) if weights else (ST2_ROOT / config["F0_path"])
    print(f"loading JDCNet from {f0_path}", flush=True)
    pe = load_F0_models(str(f0_path))
    pe.to(device)
    pe.train()
    for p in pe.parameters():
        p.requires_grad_(True)
    return pe


def save_jdc_checkpoint(model, path: Path, meta: dict | None = None) -> None:
    """Save in StyleTTS2 bst.t7 layout: {'net': state_dict, ...}."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"net": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
    if meta:
        payload["meta"] = meta
    torch.save(payload, path)
    print(f"saved {path}", flush=True)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
