"""
Build train/dev CSV manifests from LibriTTS_R.

  python -m timbre_encoder.build_librispeech_manifest
"""
from __future__ import annotations

import argparse
import csv
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "dataset"
DEFAULT_OUT = ROOT / "timbre_encoder" / "manifests"


def extract_archives(dataset_dir: Path) -> Path:
    libritts = dataset_dir / "LibriTTS_R"
    archives = {
        "dev-clean": dataset_dir / "dev_clean.tar.gz",
        "train-clean-100": dataset_dir / "train_clean_100.tar.gz",
    }
    for split, tar_path in archives.items():
        split_dir = libritts / split
        if split_dir.is_dir() and any(split_dir.glob("*")):
            print(f"[skip] {split} present", flush=True)
            continue
        if not tar_path.is_file():
            print(f"[warn] missing {tar_path}", flush=True)
            continue
        print(f"[extract] {tar_path.name}", flush=True)
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(dataset_dir)
        print(f"[done] {split}", flush=True)
    return libritts


def build_split_manifest(split_dir: Path, out_csv: Path) -> int:
    print(f"[scan] {split_dir}", flush=True)
    rows = []
    count = 0
    for wav in split_dir.rglob("*.wav"):
        count += 1
        if count % 5000 == 0:
            print(f"  ... scanned {count} wavs", flush=True)
        stem = wav.stem
        text = None
        for suffix in (".normalized.txt", ".original.txt"):
            candidate = wav.parent / f"{stem}{suffix}"
            if candidate.is_file():
                text = candidate.read_text(encoding="utf-8", errors="ignore").strip()
                break
        if not text:
            continue
        # path: split/SPEAKER/CHAPTER/file.wav
        rel = wav.relative_to(split_dir)
        parts = rel.parts
        speaker_id = parts[0] if len(parts) > 1 else "unknown"
        chapter_id = parts[1] if len(parts) > 2 else "unknown"
        rows.append(
            {
                "audio_path": str(wav),
                "text": text,
                "speaker_id": speaker_id,
                "chapter_id": chapter_id,
                "duration_sec": "",
            }
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["audio_path", "text", "speaker_id", "chapter_id", "duration_sec"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[manifest] {out_csv.name}: {len(rows)} rows (from {count} wavs)", flush=True)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_DATASET / "LibriTTS_R")
    ap.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--splits", nargs="+", default=["dev-clean", "train-clean-100"])
    args = ap.parse_args()

    if args.extract:
        args.root = extract_archives(args.dataset_dir)

    total = 0
    for split in args.splits:
        split_dir = args.root / split
        if not split_dir.is_dir():
            print(f"[skip] missing {split_dir}", flush=True)
            continue
        total += build_split_manifest(split_dir, args.out_dir / f"{split}.csv")
    print(f"[done] total={total}", flush=True)


if __name__ == "__main__":
    main()
