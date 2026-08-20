#!/usr/bin/env python3
"""QC LibriTTS_R + kokoro_targets_v1 and write canonical train/dev indexes.

Does not modify phoneme strings or Kokoro IDs (no Y->y / ʏ->y remaps).
Keys rows by audio path relative to --data-root.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_NPZ = ("f0", "n", "d_gt", "phoneme_ids", "mel_len")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            rows.append(value)
    return rows


def index_by_audio(rows: list[dict[str, Any]], key: str = "audio") -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        audio = str(row.get(key) or "").replace("\\", "/").lstrip("./")
        if not audio:
            raise ValueError(f"row missing {key}: {row}")
        if audio in out:
            raise ValueError(f"duplicate audio key: {audio}")
        out[audio] = row
    return out


def rel_audio_from_csv(audio_path: str, data_root: Path) -> str:
    raw = audio_path.strip().replace("\\", "/")
    root = data_root.resolve().as_posix().rstrip("/")
    path = Path(os.path.expandvars(os.path.expanduser(raw)))
    if path.is_absolute():
        abs_posix = path.resolve().as_posix()
        if abs_posix.lower().startswith(root.lower() + "/"):
            return abs_posix[len(root) + 1 :]
        # Fallback: keep last 4 LibriTTS components: split/spk/chap/utt.wav
        parts = Path(abs_posix).parts
        if len(parts) >= 4:
            return "/".join(parts[-4:])
        return Path(abs_posix).name
    return raw.lstrip("./")


def audit_npz(npz_path: Path) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=False) as data:
        missing = [k for k in REQUIRED_NPZ if k not in data.files]
        if missing:
            raise ValueError(f"NPZ missing {missing}")
        if "stage" in data.files and str(data["stage"]) == "jdc":
            raise ValueError("partial JDC-only NPZ (no d_gt)")
        f0 = np.asarray(data["f0"]).reshape(-1)
        n = np.asarray(data["n"]).reshape(-1)
        d_gt = np.asarray(data["d_gt"]).reshape(-1)
        ids = np.asarray(data["phoneme_ids"]).reshape(-1).astype(np.int64)
        mel_len = int(np.asarray(data["mel_len"]).item())
        align_score = float(np.asarray(data["align_score"]).item()) if "align_score" in data.files else float("nan")
        phonemes = str(data["phonemes"].item()) if "phonemes" in data.files else ""
        text = str(data["text"].item()) if "text" in data.files else ""

    if len(f0) != mel_len or len(n) != mel_len:
        raise ValueError(f"len(f0/n) != mel_len ({len(f0)},{len(n)},{mel_len})")
    if len(d_gt) != len(ids):
        raise ValueError(f"len(d_gt)={len(d_gt)} != len(ids)={len(ids)}")
    if int(np.asarray(d_gt).sum()) != mel_len:
        raise ValueError(f"sum(d_gt)={int(d_gt.sum())} != mel_len={mel_len}")
    if len(ids) == 0 or len(ids) > 510:
        raise ValueError(f"phoneme_ids length {len(ids)} out of range")
    if np.any(ids < 0) or np.any(ids >= 178):
        raise ValueError("phoneme_ids outside Kokoro vocab [0,177]")
    if not np.all(np.isfinite(f0)) or not np.all(np.isfinite(n)):
        raise ValueError("non-finite f0/n")
    if np.any(d_gt < 1):
        raise ValueError("d_gt must be >= 1 per token")
    return {
        "phoneme_ids": ids.tolist(),
        "mel_len": mel_len,
        "align_score": align_score,
        "phonemes": phonemes,
        "text": text,
        "phoneme_len": int(len(ids)),
    }


def prepare_split(
    split: str,
    csv_path: Path,
    phonemes_map: dict[str, dict[str, Any]],
    targets_map: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    rejected: Counter[str],
) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for csv_row in reader:
            audio_abs = csv_row.get("audio_path") or ""
            if not audio_abs:
                rejected["missing_audio_path"] += 1
                continue
            rel = rel_audio_from_csv(audio_abs, args.data_root)
            if str(csv_row.get("align_status", "")).lower() != "ok":
                rejected["align_status"] += 1
                continue
            ph = phonemes_map.get(rel)
            tgt = targets_map.get(rel)
            if ph is None:
                rejected["missing_phonemes"] += 1
                continue

            if tgt is None:
                # Fallback: NPZ mirrors audio relative path
                cache_rel = str(Path(rel).with_suffix(".npz")).replace("\\", "/")
                npz_guess = (args.targets_root / cache_rel).resolve()
                if npz_guess.is_file():
                    tgt = {
                        "audio": rel,
                        "cache": cache_rel,
                        "speaker_id": csv_row.get("speaker_id"),
                    }
                else:
                    rejected["missing_target_manifest"] += 1
                    continue

            cache_rel = str(tgt.get("cache") or Path(rel).with_suffix(".npz")).replace("\\", "/")
            npz_path = (args.targets_root / cache_rel).resolve()
            wav_path = (args.data_root / rel).resolve()
            if not wav_path.is_file():
                rejected["missing_audio"] += 1
                continue
            if not npz_path.is_file():
                rejected["missing_npz"] += 1
                continue

            try:
                audited = audit_npz(npz_path)
            except (OSError, ValueError) as exc:
                rejected[f"invalid_npz:{type(exc).__name__}"] += 1
                continue

            score = audited["align_score"]
            if np.isnan(score):
                try:
                    score = float(tgt.get("align_score", float("nan")))
                except (TypeError, ValueError):
                    score = float("nan")
            if np.isnan(score) or score < args.min_align_score:
                rejected["align_score"] += 1
                continue

            speaker = str(csv_row.get("speaker_id") or tgt.get("speaker_id") or "0")
            phonemes = str(ph.get("phonemes") or audited["phonemes"] or "")
            # Exact Kokoro IDs from NPZ — no string remaps.
            accepted.append(
                {
                    "id": Path(rel).stem,
                    "split": split,
                    "audio": rel,
                    "cache": cache_rel,
                    "audio_path": wav_path.as_posix(),
                    "npz_path": npz_path.as_posix(),
                    "speaker": speaker,
                    "speaker_id": int(speaker),
                    "text": str(ph.get("text") or audited["text"] or csv_row.get("text") or ""),
                    "phonemes": phonemes,
                    "phoneme_ids": audited["phoneme_ids"],
                    "align_score": float(score),
                    "mel_len": audited["mel_len"],
                    "phoneme_len": audited["phoneme_len"],
                }
            )
    return sorted(accepted, key=lambda row: row["audio"])


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    phonemes_map = index_by_audio(read_jsonl(args.phonemes_jsonl), "audio")
    targets_map = index_by_audio(read_jsonl(args.targets_manifest), "audio")
    rejected: Counter[str] = Counter()

    splits = {
        "train": prepare_split("train", args.train_csv, phonemes_map, targets_map, args, rejected),
        "dev": prepare_split("dev", args.dev_csv, phonemes_map, targets_map, args, rejected),
    }
    train_ids = {r["id"] for r in splits["train"]}
    dev_ids = {r["id"] for r in splits["dev"]}
    if train_ids & dev_ids:
        raise ValueError(f"train/dev id overlap: {sorted(train_ids & dev_ids)[:5]}")

    train_speakers = {r["speaker"] for r in splits["train"]}
    dev_speakers = {r["speaker"] for r in splits["dev"]}
    if train_speakers & dev_speakers:
        raise ValueError(f"train/dev speaker overlap: {sorted(train_speakers & dev_speakers)[:5]}")

    # Stable speaker map: train first, then dev (contiguous ints for logging only;
    # StyleTTS2 list uses native LibriTTS speaker ids).
    speakers = sorted(train_speakers, key=lambda s: int(s)) + sorted(dev_speakers, key=lambda s: int(s))
    speaker_map = {spk: int(spk) for spk in speakers}

    for split, rows in splits.items():
        jsonl_path = args.output_dir / f"{split}.jsonl"
        list_path = args.output_dir / f"{split}_list.txt"
        with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        with list_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                # wav|phonemes|speaker — relative to data_root for StyleTTS2 root_path
                handle.write(f"{row['audio']}|{row['phonemes']}|{row['speaker_id']}\n")
        write_json(args.output_dir / f"{split}.index.json", {row["id"]: i for i, row in enumerate(rows)})

    ood = [r["phonemes"] for r in splits["dev"] if r["phonemes"]]
    (args.output_dir / "OOD_texts.txt").write_text("\n".join(ood) + ("\n" if ood else ""), encoding="utf-8")
    write_json(args.output_dir / "speaker_map.json", speaker_map)

    audit = {
        "schema": "kokoro_runpod_audit_v1",
        "accepted": {k: len(v) for k, v in splits.items()},
        "speakers": {"train": len(train_speakers), "dev": len(dev_speakers)},
        "rejected": dict(sorted(rejected.items())),
        "settings": {
            "min_align_score": args.min_align_score,
            "max_phonemes": 510,
            "phoneme_remaps": False,
        },
    }
    write_json(args.output_dir / "audit_summary.json", audit)
    if not splits["train"] or not splits["dev"]:
        raise ValueError("empty train or dev after QC")
    return audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-csv", type=Path, required=True)
    p.add_argument("--dev-csv", type=Path, required=True)
    p.add_argument("--phonemes-jsonl", type=Path, required=True)
    p.add_argument("--targets-manifest", type=Path, required=True)
    p.add_argument("--targets-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True, help="LibriTTS_R root")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--min-align-score", type=float, default=0.25)
    args = p.parse_args(argv)
    for name in ("train_csv", "dev_csv", "phonemes_jsonl", "targets_manifest", "targets_root", "data_root", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        audit = run(parse_args(argv))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
