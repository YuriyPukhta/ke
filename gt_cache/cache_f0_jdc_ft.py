"""
Cache fine-tuned JDCNet F0 (Kokoro units) for LibriTTS manifests.

Writes float32 .npy under gt_cache/features/f0_jdc_ft/...
and updates CSV columns: f0_jdc_ft_status, f0_jdc_ft_len.

Does not overwrite baseline JDC features/f0.

Usage:
  .venv\\Scripts\\python.exe -m gt_cache.cache_f0_jdc_ft \\
      --csv timbre_encoder/manifests/dev-clean.filtered.csv

  .venv\\Scripts\\python.exe -m gt_cache.cache_f0_jdc_ft \\
      --csv timbre_encoder/manifests/train-clean-100.filtered.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gt_cache.jdc_finetune import DEFAULT_CKPT_DIR  # noqa: E402
from gt_cache.utils import (  # noqa: E402
    DEFAULT_FEATURES_ROOT,
    F0_JDC_FT_FIELDS,
    ensure_fields,
    extract_f0,
    feature_path,
    load_pitch_extractor,
    load_wav_for_cache,
    save_curve,
    wav_to_mel,
    write_csv,
)

DEFAULT_WEIGHTS = DEFAULT_CKPT_DIR / "jdc_kokoro_best.t7"
KIND = "f0_jdc_ft"


def needs_f0(row: dict[str, str], features_root: Path, force: bool) -> bool:
    if force:
        return True
    if row.get("f0_jdc_ft_status") != "ok":
        return True
    path = feature_path(row.get("audio_path", ""), KIND, features_root)
    return not path.is_file()


def process_row(
    row: dict[str, str],
    pitch_extractor,
    device: torch.device,
    features_root: Path,
    max_wav_sec: float,
) -> dict[str, str]:
    out = dict(row)
    for k in F0_JDC_FT_FIELDS:
        out.setdefault(k, "")

    audio = row.get("audio_path", "")
    wav = load_wav_for_cache(audio, max_wav_sec=max_wav_sec)
    if wav is None:
        out["f0_jdc_ft_status"] = "missing_or_short_audio"
        out["f0_jdc_ft_len"] = ""
        return out

    try:
        mel = wav_to_mel(wav, device=device)
        f0 = extract_f0(pitch_extractor, mel)
        if f0.numel() != mel.shape[-1]:
            out["f0_jdc_ft_status"] = f"len_mismatch:{f0.numel()}!={mel.shape[-1]}"
            out["f0_jdc_ft_len"] = str(int(f0.numel()))
            return out
        out_path = feature_path(audio, KIND, features_root)
        save_curve(out_path, f0)
        out["f0_jdc_ft_status"] = "ok"
        out["f0_jdc_ft_len"] = str(int(f0.numel()))
    except Exception as e:
        out["f0_jdc_ft_status"] = f"fail:{type(e).__name__}"
        out["f0_jdc_ft_len"] = ""
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Cache fine-tuned JDC F0 into features/f0_jdc_ft")
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--features-root", type=str, default=str(DEFAULT_FEATURES_ROOT))
    ap.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--speaker", type=str, default="", help="optional speaker_id filter")
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--max-wav-sec", type=float, default=8.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise SystemExit(f"missing csv: {csv_path}")
    weights = Path(args.weights)
    if not weights.is_file():
        raise SystemExit(f"missing weights: {weights}")
    features_root = Path(args.features_root)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(
        f"device={device} csv={csv_path} features={features_root} weights={weights}",
        flush=True,
    )

    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = ensure_fields(list(reader.fieldnames or []), F0_JDC_FT_FIELDS)

    pitch_extractor = load_pitch_extractor(device, weights=weights)
    pending = [i for i, r in enumerate(rows) if needs_f0(r, features_root, args.force)]
    if args.speaker:
        pending = [i for i in pending if rows[i].get("speaker_id") == args.speaker]
    if args.limit > 0:
        pending = pending[: args.limit]
    print(
        f"rows={len(rows)} to_process={len(pending)} "
        f"speaker={args.speaker or '-'} force={args.force}",
        flush=True,
    )

    processed = 0
    for n, idx in enumerate(pending, 1):
        rows[idx] = process_row(
            rows[idx],
            pitch_extractor,
            device=device,
            features_root=features_root,
            max_wav_sec=args.max_wav_sec,
        )
        processed += 1
        status = rows[idx].get("f0_jdc_ft_status", "")
        if n % 20 == 0 or status != "ok":
            print(
                f"[{n}/{len(pending)}] idx={idx} status={status} "
                f"len={rows[idx].get('f0_jdc_ft_len')}",
                flush=True,
            )
        if processed % args.save_every == 0:
            write_csv(csv_path, fieldnames, rows)
            print(f"  checkpoint saved ({processed} new)", flush=True)

    write_csv(csv_path, fieldnames, rows)
    ok = sum(1 for r in rows if r.get("f0_jdc_ft_status") == "ok")
    print(f"done f0_jdc_ft_ok={ok}/{len(rows)} wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
