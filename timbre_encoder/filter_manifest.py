"""
Filter LibriTTS_R manifests by SNR (book.tsv), English text, and excluded speakers.

Usage:
  python -m timbre_encoder.filter_manifest \\
    --in-csv timbre_encoder/manifests/train-clean-100.csv \\
    --out-csv timbre_encoder/manifests/train-clean-100.filtered.csv \\
    --min-snr 20
"""
from __future__ import annotations

import argparse
import csv
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# LibriTTS-P speakers with multiple speakers detected in one ID
DEFAULT_EXCLUDED_SPEAKERS = {
    "2074",
    "4455",
    "6032",
    "3546",
    "2262",
    "8097",
    "1734",
    "3793",
    "8295",
}

# Non-English / non-Latin scripts
_NON_EN_RE = re.compile(
    r"[\u0400-\u04FF\u0500-\u052F\u2DE0-\u2DFF\uA640-\uA69F"  # Cyrillic
    r"\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF"  # JP/CN
    r"\uAC00-\uD7AF"  # Korean
    r"\u0600-\u06FF\u0750-\u077F"  # Arabic
    r"\u0590-\u05FF]"  # Hebrew
)


def load_snr_index(split_dir: Path) -> dict[str, float]:
    """utt_id -> SNR from *.book.tsv last column."""
    snr_map: dict[str, float] = {}
    for book_tsv in split_dir.rglob("*.book.tsv"):
        for line in book_tsv.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split("\t")
            if len(parts) < 7:
                continue
            utt_id = parts[0].strip()
            try:
                snr_map[utt_id] = float(parts[-1])
            except ValueError:
                continue
    return snr_map


def is_english_text(text: str) -> bool:
    if not text or not text.strip():
        return False
    if _NON_EN_RE.search(text):
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    basic = sum(1 for c in letters if ("A" <= c <= "Z") or ("a" <= c <= "z"))
    return basic / len(letters) >= 0.85


def utt_id_from_path(audio_path: str) -> str:
    return Path(audio_path).stem


def fetch_excluded_speakers(url: str | None) -> set[str]:
    speakers = set(DEFAULT_EXCLUDED_SPEAKERS)
    if not url:
        return speakers
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                speakers.add(line.split()[0])
    except Exception as e:
        print(f"[warn] could not fetch excluded speakers: {e}", flush=True)
    return speakers


def filter_csv(
    in_csv: Path,
    out_csv: Path,
    data_root: Path,
    min_snr: float,
    excluded: set[str],
    split_name: str | None = None,
) -> tuple[int, int]:
    rows_in = list(csv.DictReader(in_csv.open(encoding="utf-8")))
    if not rows_in:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            f.write("audio_path,text,speaker_id,chapter_id,duration_sec,snr\n")
        return 0, 0

    # Infer split dir from first path or explicit name
    sample = Path(rows_in[0]["audio_path"])
    # .../LibriTTS_R/train-clean-100/SPEAKER/...
    split_dir = None
    for parent in sample.parents:
        if parent.name.startswith("train-") or parent.name.startswith("dev-") or parent.name.startswith("test-"):
            split_dir = parent
            break
    if split_dir is None and split_name:
        split_dir = data_root / split_name
    if split_dir is None or not split_dir.is_dir():
        raise FileNotFoundError(f"Cannot resolve split dir for {in_csv}")

    print(f"[snr] indexing {split_dir} ...", flush=True)
    snr_map = load_snr_index(split_dir)
    print(f"[snr] {len(snr_map)} utterances with SNR", flush=True)

    kept: list[dict] = []
    stats = {"no_snr": 0, "low_snr": 0, "bad_spk": 0, "non_en": 0, "ok": 0}
    for row in rows_in:
        utt = utt_id_from_path(row["audio_path"])
        spk = str(row.get("speaker_id", ""))
        if spk in excluded:
            stats["bad_spk"] += 1
            continue
        if not is_english_text(row.get("text", "")):
            stats["non_en"] += 1
            continue
        if utt not in snr_map:
            stats["no_snr"] += 1
            continue
        snr = snr_map[utt]
        if snr < min_snr:
            stats["low_snr"] += 1
            continue
        row = dict(row)
        row["snr"] = f"{snr:.4f}"
        kept.append(row)
        stats["ok"] += 1

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["audio_path", "text", "speaker_id", "chapter_id", "duration_sec", "snr"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(kept)

    print(
        f"[filter] {in_csv.name}: {len(rows_in)} -> {len(kept)} "
        f"(min_snr={min_snr}) | {stats}",
        flush=True,
    )
    return len(rows_in), len(kept)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", type=Path, action="append", dest="inputs")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "timbre_encoder" / "manifests")
    ap.add_argument("--data-root", type=Path, default=ROOT / "dataset" / "LibriTTS_R")
    ap.add_argument("--min-snr", type=float, default=20.0)
    ap.add_argument(
        "--excluded-url",
        default="https://raw.githubusercontent.com/line/LibriTTS-P/main/data/excluded_spk_list.txt",
    )
    ap.add_argument("--all-defaults", action="store_true", help="Filter train+dev clean manifests")
    args = ap.parse_args()

    excluded = fetch_excluded_speakers(args.excluded_url)
    print(f"[excluded speakers] {sorted(excluded)}", flush=True)

    inputs = args.inputs or []
    if args.all_defaults or not inputs:
        inputs = [
            ROOT / "timbre_encoder" / "manifests" / "dev-clean.csv",
            ROOT / "timbre_encoder" / "manifests" / "train-clean-100.csv",
        ]

    for in_csv in inputs:
        in_csv = Path(in_csv)
        if not in_csv.is_file():
            print(f"[skip] missing {in_csv}", flush=True)
            continue
        out_csv = args.out_dir / (in_csv.stem + ".filtered.csv")
        filter_csv(in_csv, out_csv, args.data_root, args.min_snr, excluded)


if __name__ == "__main__":
    main()
