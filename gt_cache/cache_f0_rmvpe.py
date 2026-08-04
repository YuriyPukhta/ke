"""
Cache RMVPE F0 curves for LibriTTS manifests.

Writes float32 .npy under gt_cache/features/f0_rmvpe/... (Hz, resized to mel frames),
and updates CSV columns: f0_rmvpe_status, f0_rmvpe_len.

Does not overwrite JDC features/f0.

Usage:
  .venv\\Scripts\\python.exe -m gt_cache.cache_f0_rmvpe \\
      --csv timbre_encoder/manifests/train-clean-100.filtered.csv --speaker 6209
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gt_cache.rmvpe_backend import extract_f0_rmvpe, load_rmvpe  # noqa: E402
from gt_cache.utils import (  # noqa: E402
    DEFAULT_FEATURES_ROOT,
    F0_RMVPE_FIELDS,
    ensure_fields,
    feature_path,
    load_wav_for_cache,
    save_curve,
    wav_to_mel,
    write_csv,
)


def needs_f0(row: dict[str, str], features_root: Path, force: bool) -> bool:
    if force:
        return True
    if row.get("f0_rmvpe_status") != "ok":
        return True
    path = feature_path(row.get("audio_path", ""), "f0_rmvpe", features_root)
    return not path.is_file()


def process_row(
    row: dict[str, str],
    rmvpe,
    device: torch.device,
    features_root: Path,
    max_wav_sec: float,
    thred: float,
    smooth: bool = False,
    max_uv_gap: int = 16,
    median_k: int = 7,
    edge_fade: int = 4,
) -> dict[str, str]:
    out = dict(row)
    for k in F0_RMVPE_FIELDS:
        out.setdefault(k, "")

    audio = row.get("audio_path", "")
    wav = load_wav_for_cache(audio, max_wav_sec=max_wav_sec)
    if wav is None:
        out["f0_rmvpe_status"] = "missing_or_short_audio"
        out["f0_rmvpe_len"] = ""
        return out

    try:
        mel = wav_to_mel(wav, device=device)
        n_frames = int(mel.shape[-1])
        f0 = extract_f0_rmvpe(
            rmvpe,
            wav,
            n_frames,
            thred=thred,
            smooth=smooth,
            max_uv_gap=max_uv_gap,
            median_k=median_k,
            edge_fade=edge_fade,
        )
        if f0.numel() != n_frames:
            out["f0_rmvpe_status"] = f"len_mismatch:{f0.numel()}!={n_frames}"
            out["f0_rmvpe_len"] = str(int(f0.numel()))
            return out
        out_path = feature_path(audio, "f0_rmvpe", features_root)
        save_curve(out_path, f0)
        out["f0_rmvpe_status"] = "ok"
        out["f0_rmvpe_len"] = str(int(f0.numel()))
    except Exception as e:
        out["f0_rmvpe_status"] = f"fail:{type(e).__name__}"
        out["f0_rmvpe_len"] = ""
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Cache RMVPE F0 into features/f0_rmvpe + manifest")
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--features-root", type=str, default=str(DEFAULT_FEATURES_ROOT))
    ap.add_argument("--model", type=str, default="", help="path to rmvpe.pt")
    ap.add_argument("--speaker", type=str, default="", help="optional speaker_id filter")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--max-wav-sec", type=float, default=8.0)
    ap.add_argument("--thred", type=float, default=0.01)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--fp32", action="store_true", help="disable fp16 even on CUDA")
    ap.add_argument("--smooth", action="store_true", help="fill UV + median + edge fade")
    ap.add_argument("--max-uv-gap", type=int, default=16)
    ap.add_argument("--median-k", type=int, default=7)
    ap.add_argument("--edge-fade", type=int, default=4)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise SystemExit(f"missing csv: {csv_path}")
    features_root = Path(args.features_root)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(
        f"device={device} csv={csv_path} features={features_root} smooth={args.smooth}",
        flush=True,
    )

    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = ensure_fields(list(reader.fieldnames or []), F0_RMVPE_FIELDS)

    if args.speaker:
        candidate_idx = [
            i for i, r in enumerate(rows) if r.get("speaker_id") == args.speaker
        ]
    else:
        candidate_idx = list(range(len(rows)))

    pending = [
        i for i in candidate_idx if needs_f0(rows[i], features_root, args.force)
    ]
    if args.limit > 0:
        pending = pending[: args.limit]
    print(
        f"rows={len(rows)} speaker_filter={args.speaker or '-'} "
        f"to_process={len(pending)} force={args.force}",
        flush=True,
    )
    if not pending:
        print("nothing to do", flush=True)
        return

    rmvpe, ckpt = load_rmvpe(
        device,
        model_path=args.model or None,
        is_half=(device.type == "cuda" and not args.fp32),
    )
    print(f"ckpt={ckpt}", flush=True)

    processed = 0
    for n, idx in enumerate(pending, 1):
        rows[idx] = process_row(
            rows[idx],
            rmvpe,
            device=device,
            features_root=features_root,
            max_wav_sec=args.max_wav_sec,
            thred=args.thred,
            smooth=args.smooth,
            max_uv_gap=args.max_uv_gap,
            median_k=args.median_k,
            edge_fade=args.edge_fade,
        )
        processed += 1
        status = rows[idx].get("f0_rmvpe_status", "")
        if n % 20 == 0 or status != "ok":
            print(
                f"[{n}/{len(pending)}] idx={idx} status={status} "
                f"len={rows[idx].get('f0_rmvpe_len')}",
                flush=True,
            )
        if processed % args.save_every == 0:
            write_csv(csv_path, fieldnames, rows)
            print(f"  checkpoint saved ({processed} new)", flush=True)

    write_csv(csv_path, fieldnames, rows)
    ok = sum(1 for r in rows if r.get("f0_rmvpe_status") == "ok")
    print(f"done f0_rmvpe_ok={ok}/{len(rows)} wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
