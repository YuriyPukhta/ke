"""
Cache d_gt with fine-tuned TMA (Kokoro duration distill).

Writes CSV columns (does NOT overwrite baseline d_gt / align_status):
  d_gt_tma_ft, mel_down_len_tma_ft, align_status_tma_ft

Reuses phonemes_st2 from the manifest when present; otherwise normalizes like cache_d_gt.

Usage:
  .venv\\Scripts\\python.exe -m gt_cache.cache_d_gt_tma_ft \\
      --csv timbre_encoder/manifests/dev-clean.filtered.csv

  .venv\\Scripts\\python.exe -m gt_cache.cache_d_gt_tma_ft \\
      --csv timbre_encoder/manifests/train-clean-100.filtered.csv --force
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kokoro.pipeline import KPipeline  # noqa: E402

from gt_cache.cache_d_gt import encode_d_gt  # noqa: E402
from gt_cache.tma_finetune import DEFAULT_CKPT_DIR  # noqa: E402
from gt_cache.utils import write_csv  # noqa: E402
from timbre_encoder.aligner import FrozenTextAligner, TextCleaner  # noqa: E402
from timbre_encoder.librispeech_dataset import text_to_kokoro_phonemes  # noqa: E402
from timbre_encoder.mel_preprocess import SAMPLE_RATE, waveform_to_mel  # noqa: E402
from timbre_encoder.phoneme_norm import normalize_kokoro_phonemes_for_st2, norm_summary  # noqa: E402

DEFAULT_WEIGHTS = DEFAULT_CKPT_DIR / "tma_kokoro_best.pth"

FT_FIELDS = (
    "d_gt_tma_ft",
    "mel_down_len_tma_ft",
    "align_status_tma_ft",
)


def needs_ft(row: dict[str, str], force: bool) -> bool:
    if force:
        return True
    if row.get("align_status_tma_ft") != "ok":
        return True
    if not (row.get("d_gt_tma_ft") or "").strip():
        return True
    ph = (row.get("phonemes_st2") or "").strip()
    if ph and len(row["d_gt_tma_ft"].split()) != len(ph):
        return True
    return False


def resolve_phonemes_st2(row: dict[str, str], pipeline) -> tuple[str, dict[str, str]]:
    """Return (phonemes_st2, optional column updates for phonemes_*)."""
    updates: dict[str, str] = {}
    ph_st2 = (row.get("phonemes_st2") or "").strip()
    if ph_st2:
        return ph_st2, updates

    text = (row.get("text") or "").strip()
    raw = (row.get("phonemes") or "").strip()
    if not raw:
        if not text:
            return "", updates
        raw = text_to_kokoro_phonemes(text, pipeline)
    nr = normalize_kokoro_phonemes_for_st2(raw)
    updates["phonemes"] = nr.raw
    updates["phonemes_st2"] = nr.normalized
    updates["norm_applied"] = "1" if nr.changed else "0"
    updates["norm_ops"] = norm_summary(nr)
    updates["len_phonemes"] = str(nr.len_raw)
    updates["len_phonemes_st2"] = str(nr.len_norm)
    updates["delta_len"] = str(nr.delta_len)
    return nr.normalized, updates


def process_row(
    row: dict[str, str],
    pipeline,
    aligner: FrozenTextAligner,
    max_wav_sec: float,
    min_phonemes: int,
    max_phonemes: int,
) -> dict[str, str]:
    out = dict(row)
    for k in FT_FIELDS:
        out.setdefault(k, "")

    path = row.get("audio_path", "")
    if not path or not Path(path).is_file():
        out["align_status_tma_ft"] = "missing_audio"
        return out

    try:
        ph_st2, updates = resolve_phonemes_st2(row, pipeline)
        out.update(updates)
    except Exception as e:
        out["align_status_tma_ft"] = f"g2p_fail:{type(e).__name__}"
        return out

    if not ph_st2:
        out["align_status_tma_ft"] = "empty_phonemes"
        return out
    n_ph = len(ph_st2)
    if not (min_phonemes <= n_ph <= max_phonemes):
        out["align_status_tma_ft"] = f"phoneme_len:{n_ph}"
        return out

    try:
        wav, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        wav, _ = librosa.effects.trim(wav, top_db=30)
        max_samples = int(max_wav_sec * SAMPLE_RATE)
        if wav.shape[0] > max_samples:
            wav = wav[:max_samples]
        if wav.shape[0] < SAMPLE_RATE // 4:
            out["align_status_tma_ft"] = "wav_too_short"
            return out
        mel = waveform_to_mel(
            torch.from_numpy(wav.astype(np.float32)), device=aligner._device
        ).squeeze(0)

        a_gt, d_gt, _ = aligner.align(mel, ph_st2)
        mel_down = int(a_gt.shape[-1])
        d_list = [max(0, int(round(float(x)))) for x in d_gt.tolist()]

        if len(d_list) != n_ph:
            out["align_status_tma_ft"] = f"d_gt_len_mismatch:{len(d_list)}!={n_ph}"
            return out
        if sum(d_list) != mel_down:
            out["align_status_tma_ft"] = "mas_sum_mismatch"
            return out
        if len(TextCleaner()(ph_st2)) != n_ph:
            out["align_status_tma_ft"] = "st2_still_drops"
            return out

        out["d_gt_tma_ft"] = encode_d_gt(d_gt)
        out["mel_down_len_tma_ft"] = str(mel_down)
        out["align_status_tma_ft"] = "ok"
    except Exception as e:
        out["align_status_tma_ft"] = f"align_fail:{type(e).__name__}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Cache d_gt with fine-tuned TMA")
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_WEIGHTS),
        help="Fine-tuned ASRCNN checkpoint",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--max-wav-sec", type=float, default=8.0)
    ap.add_argument("--min-phonemes", type=int, default=2)
    ap.add_argument("--max-phonemes", type=int, default=400)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    weights = Path(args.weights)
    if not csv_path.is_file():
        raise SystemExit(f"missing csv: {csv_path}")
    if not weights.is_file():
        raise SystemExit(f"missing weights: {weights}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device} csv={csv_path} weights={weights}", flush=True)

    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    for col in FT_FIELDS:
        if col not in fieldnames:
            fieldnames.append(col)

    print(f"rows={len(rows)} loading G2P + fine-tuned TMA...", flush=True)
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    aligner = FrozenTextAligner(checkpoint=weights, device=device)

    pending = [i for i, r in enumerate(rows) if needs_ft(r, force=args.force)]
    already_ok = len(rows) - len(pending)
    if args.limit > 0:
        pending = pending[: args.limit]
    print(
        f"already_ok_skip={already_ok} to_process={len(pending)} force={args.force}",
        flush=True,
    )

    processed = 0
    for n, idx in enumerate(pending, 1):
        rows[idx] = process_row(
            rows[idx],
            pipeline,
            aligner,
            max_wav_sec=args.max_wav_sec,
            min_phonemes=args.min_phonemes,
            max_phonemes=args.max_phonemes,
        )
        processed += 1
        status = rows[idx].get("align_status_tma_ft", "")
        if n % 20 == 0 or status != "ok":
            print(f"[{n}/{len(pending)}] idx={idx} status={status}", flush=True)
        if processed % args.save_every == 0:
            write_csv(csv_path, fieldnames, rows)
            print(f"  checkpoint saved ({processed} new)", flush=True)

    write_csv(csv_path, fieldnames, rows)
    ok = sum(1 for r in rows if r.get("align_status_tma_ft") == "ok")
    print(f"done ok={ok}/{len(rows)} wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
