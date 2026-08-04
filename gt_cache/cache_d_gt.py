"""
Cache phoneme GT alignment (d_gt) into filtered manifests.

Kokoro/Misaki phonemes stay in `phonemes`. ST2-facing copies go to *new* columns;
alignment always runs on `phonemes_st2` so len(d_gt) == len(phonemes_st2).

Columns:
  phonemes          — raw Kokoro/Misaki (unchanged meaning)
  phonemes_st2      — normalized for ST2 TextCleaner / aligner
  norm_applied      — 1 if phonemes != phonemes_st2 else 0
  norm_ops          — e.g. repl=2|drop=1|dlen=-1
  len_phonemes      — len(phonemes)
  len_phonemes_st2  — len(phonemes_st2)
  delta_len         — len_phonemes_st2 - len_phonemes
  d_gt, mel_len, mel_down_len, align_status

Usage:
  .venv\\Scripts\\python.exe -m gt_cache.cache_d_gt \\
      --csv timbre_encoder/manifests/dev-clean.filtered.csv --limit 20

  .venv\\Scripts\\python.exe -m gt_cache.cache_d_gt \\
      --csv timbre_encoder/manifests/dev-clean.filtered.csv --renorm-ok
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

from kokoro.pipeline import KPipeline

from timbre_encoder.aligner import FrozenTextAligner, ST2_SYMBOL_TO_ID, TextCleaner
from timbre_encoder.librispeech_dataset import text_to_kokoro_phonemes
from timbre_encoder.mel_preprocess import SAMPLE_RATE, waveform_to_mel
from timbre_encoder.phoneme_norm import normalize_kokoro_phonemes_for_st2, norm_summary
from gt_cache.utils import write_csv  # noqa: E402

BASE_FIELDS = ("audio_path", "text", "speaker_id", "chapter_id", "duration_sec", "snr")
# New / canonical align columns (phonemes = raw Kokoro)
ALIGN_FIELDS = (
    "phonemes",
    "phonemes_st2",
    "norm_applied",
    "norm_ops",
    "len_phonemes",
    "len_phonemes_st2",
    "delta_len",
    "d_gt",
    "mel_len",
    "mel_down_len",
    "align_status",
)


def encode_d_gt(d_gt: torch.Tensor) -> str:
    return " ".join(str(max(0, int(round(float(x))))) for x in d_gt.tolist())


def needs_reprocess(row: dict[str, str], renorm_ok: bool) -> bool:
    if row.get("align_status") != "ok" or not row.get("d_gt"):
        return True
    if not renorm_ok:
        return False
    # Need explicit ST2 column + flag
    if not (row.get("phonemes_st2") or "").strip():
        return True
    if row.get("norm_applied") not in ("0", "1"):
        return True
    ph_st2 = row["phonemes_st2"]
    d_n = len((row.get("d_gt") or "").split())
    if d_n != len(ph_st2):
        return True
    if any(c not in ST2_SYMBOL_TO_ID for c in ph_st2):
        return True
    return False


def process_row(
    row: dict[str, str],
    pipeline,
    aligner: FrozenTextAligner,
    max_wav_sec: float,
    min_phonemes: int,
    max_phonemes: int,
) -> dict[str, str]:
    out = dict(row)
    for k in ALIGN_FIELDS:
        out.setdefault(k, "")

    path = row.get("audio_path", "")
    text = row.get("text", "")
    if not path or not Path(path).is_file():
        out["align_status"] = "missing_audio"
        return out
    if not text.strip():
        out["align_status"] = "empty_text"
        return out

    try:
        phonemes = text_to_kokoro_phonemes(text, pipeline)
    except Exception as e:
        out["align_status"] = f"g2p_fail:{type(e).__name__}"
        return out

    nr = normalize_kokoro_phonemes_for_st2(phonemes)
    # phonemes = always raw Kokoro; ST2 copy in new column
    out["phonemes"] = nr.raw
    out["phonemes_st2"] = nr.normalized
    out["norm_applied"] = "1" if nr.changed else "0"
    out["norm_ops"] = norm_summary(nr)
    out["len_phonemes"] = str(nr.len_raw)
    out["len_phonemes_st2"] = str(nr.len_norm)
    out["delta_len"] = str(nr.delta_len)
    # Clear legacy columns if present (avoid confusion)
    out["phonemes_raw"] = ""
    out["normalized"] = ""
    out["len_raw"] = ""
    out["len_norm"] = ""

    if not (min_phonemes <= nr.len_norm <= max_phonemes):
        out["align_status"] = f"phoneme_len:{nr.len_norm}"
        return out
    if nr.len_norm < 1:
        out["align_status"] = "empty_after_norm"
        return out

    try:
        wav, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        wav, _ = librosa.effects.trim(wav, top_db=30)
        max_samples = int(max_wav_sec * SAMPLE_RATE)
        if wav.shape[0] > max_samples:
            wav = wav[:max_samples]
        if wav.shape[0] < SAMPLE_RATE // 4:
            out["align_status"] = "wav_too_short"
            return out
        mel = waveform_to_mel(torch.from_numpy(wav.astype(np.float32)), device=aligner._device)
        mel = mel.squeeze(0)

        a_gt, d_gt, _ = aligner.align(mel, nr.normalized)
        mel_len = int(mel.shape[-1])
        mel_down = int(a_gt.shape[-1])
        d_list = [max(0, int(round(float(x)))) for x in d_gt.tolist()]

        if len(d_list) != nr.len_norm:
            out["align_status"] = f"d_gt_len_mismatch:{len(d_list)}!={nr.len_norm}"
            return out
        if sum(d_list) != mel_down:
            out["align_status"] = "mas_sum_mismatch"
            return out
        if len(TextCleaner()(nr.normalized)) != nr.len_norm:
            out["align_status"] = "st2_still_drops"
            return out

        out["d_gt"] = encode_d_gt(d_gt)
        out["mel_len"] = str(mel_len)
        out["mel_down_len"] = str(mel_down)
        out["align_status"] = "ok"
    except Exception as e:
        out["align_status"] = f"align_fail:{type(e).__name__}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Cache TMA d_gt into filtered manifest CSV")
    ap.add_argument("--csv", type=str, required=True, help="Filtered manifest to update in place")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N rows needing align (0=all)")
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--max-wav-sec", type=float, default=8.0)
    ap.add_argument("--min-phonemes", type=int, default=2)
    ap.add_argument("--max-phonemes", type=int, default=400)
    ap.add_argument("--force", action="store_true", help="Recompute every row")
    ap.add_argument(
        "--renorm-ok",
        action="store_true",
        help="Recompute ok rows missing phonemes_st2 / norm_applied",
    )
    ap.add_argument(
        "--drop-legacy-cols",
        action="store_true",
        help="Omit old phonemes_raw/normalized/len_raw/len_norm from output header",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise SystemExit(f"missing csv: {csv_path}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device} csv={csv_path}", flush=True)

    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        existing = list(reader.fieldnames or [])

    legacy = {"phonemes_raw", "normalized", "len_raw", "len_norm"}
    fieldnames = list(existing)
    for col in list(BASE_FIELDS) + list(ALIGN_FIELDS):
        if col not in fieldnames:
            fieldnames.append(col)
    if args.drop_legacy_cols:
        fieldnames = [c for c in fieldnames if c not in legacy]

    if args.force:
        for r in rows:
            r["align_status"] = ""
            r["d_gt"] = ""
            r["phonemes_st2"] = ""
            r["norm_applied"] = ""

    print(f"rows={len(rows)} loading G2P + aligner...", flush=True)
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    aligner = FrozenTextAligner(device=device)

    pending = [i for i, r in enumerate(rows) if needs_reprocess(r, renorm_ok=args.renorm_ok or args.force)]
    already_ok = len(rows) - len(pending)
    if args.limit > 0:
        pending = pending[: args.limit]
    print(
        f"already_ok_skip={already_ok} to_process={len(pending)} "
        f"renorm_ok={args.renorm_ok} force={args.force}",
        flush=True,
    )

    processed = 0
    n_normed = 0
    n_delta = 0
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
        r = rows[idx]
        if r.get("norm_applied") == "1":
            n_normed += 1
        if r.get("delta_len") not in ("", "0", None):
            try:
                if int(r["delta_len"]) != 0:
                    n_delta += 1
            except ValueError:
                pass
        status = r.get("align_status", "")
        if n % 20 == 0 or status != "ok":
            print(
                f"[{n}/{len(pending)}] idx={idx} status={status} "
                f"norm={r.get('norm_applied')} dlen={r.get('delta_len')} ops={r.get('norm_ops')}",
                flush=True,
            )
        if processed % args.save_every == 0:
            write_csv(csv_path, fieldnames, rows)
            print(f"  checkpoint saved ({processed} new)", flush=True)

    write_csv(csv_path, fieldnames, rows)
    ok = sum(1 for r in rows if r.get("align_status") == "ok")
    print(
        f"done ok={ok}/{len(rows)} norm_applied_in_batch={n_normed} "
        f"delta_len_nonzero={n_delta} wrote {csv_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
