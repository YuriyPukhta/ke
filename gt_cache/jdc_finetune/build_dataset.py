"""
Build distillation pairs: Kokoro wav mel → F0Ntrain target (Kokoro units).

By default validation is **voice-held-out**: val voices never appear in train.

Usage (all voices, 10 held-out, many texts):
  .venv\\Scripts\\python.exe -m gt_cache.jdc_finetune.build_dataset \\
      --all-voices --n-val-voices 10 --limit-texts 45 \\
      --texts-csv timbre_encoder/manifests/dev-clean.filtered.csv --force
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

from kokoro.pipeline import KPipeline  # noqa: E402
from kokoro_revers.model import KModel  # noqa: E402

from gt_cache.jdc_finetune import DEFAULT_DATA_DIR  # noqa: E402
from gt_cache.jdc_finetune.common import (  # noqa: E402
    DEFAULT_TEXTS,
    DEFAULT_TRAIN_VOICES,
    DEFAULT_VAL_VOICES,
    list_kokoro_voices_from_cache,
    phonemes_from_text,
    pick_val_voices,
    pred_dur_to_alignment,
    resize_1d,
    tokens_from_phonemes,
    write_jsonl,
)
from gt_cache.utils import wav_to_mel  # noqa: E402
from timbre_encoder.optimize_speaker_ref_s import pack_slot  # noqa: E402
from timbre_encoder.st2_runtime import patch_torch_load  # noqa: E402


def load_texts_from_csv(
    csv_path: Path,
    max_n: int,
    seed: int,
    min_chars: int = 40,
    max_chars: int = 180,
) -> list[str]:
    import csv

    texts: list[str] = []
    seen: set[str] = set()
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = (row.get("text") or row.get("transcript") or "").strip()
            if len(t) < min_chars or len(t) > max_chars:
                continue
            key = t.lower()
            if key in seen:
                continue
            seen.add(key)
            texts.append(t)
    rng = random.Random(seed)
    rng.shuffle(texts)
    return texts[:max_n] if max_n > 0 else texts


def parse_voices(s: str) -> list[str]:
    return [v.strip() for v in s.split(",") if v.strip()]


def synthesize_voice(
    *,
    voice: str,
    texts: list[str],
    pipeline: KPipeline,
    model: KModel,
    device: torch.device,
    samples_dir: Path,
    out: Path,
    force: bool,
) -> tuple[list[dict], int]:
    rows: list[dict] = []
    skipped = 0
    pack = pipeline.load_voice(voice).to(device)
    if pack.dim() == 2:
        pack = pack.unsqueeze(1)

    for ti, text in enumerate(texts):
        sid = f"{voice}_{ti:04d}"
        pt_path = samples_dir / f"{sid}.pt"
        if pt_path.is_file() and not force:
            obj = torch.load(pt_path, map_location="cpu", weights_only=False)
            rows.append(
                {
                    "id": sid,
                    "path": str(pt_path.relative_to(out)).replace("\\", "/"),
                    "voice": voice,
                    "text": text,
                    "T": int(obj["f0"].numel()),
                    "status": "cached",
                }
            )
            continue

        phonemes = phonemes_from_text(pipeline, text)
        if not phonemes:
            skipped += 1
            continue
        input_ids = tokens_from_phonemes(model, phonemes)
        if not input_ids:
            skipped += 1
            continue

        ids = torch.LongTensor([[0, *input_ids, 0]]).to(device)
        ref_s = pack_slot(pack, len(phonemes)).to(device)
        audio, pred_dur = model.forward_with_tokens(ids, ref_s, speed=1.0)
        aln = pred_dur_to_alignment(pred_dur, device)
        _, f0_k, _, _ = model.forward_with_alignment(ids, ref_s, aln)
        f0_k = f0_k.detach().float().reshape(-1).cpu()

        wav = audio.detach().cpu().float().view(-1).numpy()
        mel = wav_to_mel(wav, device=device).detach().float().cpu()
        t_mel = int(mel.shape[-1])
        if t_mel < 8:
            skipped += 1
            continue
        f0 = resize_1d(f0_k, t_mel).cpu()

        torch.save(
            {
                "mel": mel.contiguous(),
                "f0": f0.contiguous(),
                "voice": voice,
                "text": text,
                "phonemes": phonemes,
            },
            pt_path,
        )
        rows.append(
            {
                "id": sid,
                "path": str(pt_path.relative_to(out)).replace("\\", "/"),
                "voice": voice,
                "text": text,
                "T": t_mel,
                "status": "ok",
            }
        )
    return rows, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description="Build JDC<-Kokoro F0 distillation dataset")
    ap.add_argument("--out", type=str, default=str(DEFAULT_DATA_DIR))
    ap.add_argument(
        "--all-voices",
        action="store_true",
        help="Use every Kokoro-82M voice from HF cache; split with --n-val-voices",
    )
    ap.add_argument("--n-val-voices", type=int, default=10, help="Held-out voices when --all-voices")
    ap.add_argument(
        "--train-voices",
        type=str,
        default=",".join(DEFAULT_TRAIN_VOICES),
        help="Voices used only for training (ignored if --all-voices)",
    )
    ap.add_argument(
        "--val-voices",
        type=str,
        default=",".join(DEFAULT_VAL_VOICES),
        help="Held-out voices (ignored if --all-voices)",
    )
    ap.add_argument(
        "--voices",
        type=str,
        default="",
        help="Optional: all voices; with --split-mode random ignores train/val-voices",
    )
    ap.add_argument(
        "--split-mode",
        type=str,
        choices=("voice", "random"),
        default="voice",
        help="voice=hold out val voices; random=shuffle rows by --val-frac",
    )
    ap.add_argument("--texts", nargs="*", default=None)
    ap.add_argument(
        "--texts-csv",
        type=str,
        default="",
        help="CSV with text/transcript (e.g. timbre_encoder/manifests/dev-clean.filtered.csv)",
    )
    ap.add_argument("--limit-texts", type=int, default=45)
    ap.add_argument("--min-chars", type=int, default=40)
    ap.add_argument("--max-chars", type=int, default=180, help="Cap text length so mel T fits large batch")
    ap.add_argument("--val-frac", type=float, default=0.15, help="Only for --split-mode random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    samples_dir = out / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.all_voices:
        all_voices = list_kokoro_voices_from_cache()
        train_voices, val_voices = pick_val_voices(all_voices, args.n_val_voices, seed=args.seed)
    else:
        train_voices = parse_voices(args.train_voices)
        val_voices = parse_voices(args.val_voices)
        if args.voices:
            all_voices = parse_voices(args.voices)
        else:
            seen: set[str] = set()
            all_voices = []
            for v in train_voices + val_voices:
                if v not in seen:
                    seen.add(v)
                    all_voices.append(v)

    overlap = set(train_voices) & set(val_voices)
    if args.split_mode == "voice" and overlap:
        raise SystemExit(f"train/val voice overlap: {sorted(overlap)}")

    if args.texts:
        texts = list(args.texts)
    elif args.texts_csv:
        texts = load_texts_from_csv(
            Path(args.texts_csv),
            args.limit_texts,
            args.seed,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
        )
    else:
        # Prefer CSV if present when building the big set without flags
        default_csv = ROOT / "timbre_encoder" / "manifests" / "dev-clean.filtered.csv"
        if default_csv.is_file() and args.limit_texts > len(DEFAULT_TEXTS):
            texts = load_texts_from_csv(
                default_csv,
                args.limit_texts,
                args.seed,
                min_chars=args.min_chars,
                max_chars=args.max_chars,
            )
        else:
            texts = list(DEFAULT_TEXTS)
            if args.limit_texts > 0:
                texts = texts[: args.limit_texts]

    if not texts:
        raise SystemExit("no texts loaded")

    print(
        f"device={device} split={args.split_mode} "
        f"n_train_voices={len(train_voices)} n_val_voices={len(val_voices)} "
        f"texts={len(texts)} expected~{len(all_voices)*len(texts)} out={out}",
        flush=True,
    )
    print(f"  train_voices={train_voices}", flush=True)
    print(f"  val_voices={val_voices}", flush=True)

    patch_torch_load()
    pipeline = KPipeline(lang_code="a", model=False)
    model = KModel(repo_id="hexgrad/Kokoro-82M").to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    rows: list[dict] = []
    skipped = 0
    with torch.no_grad():
        for vi, voice in enumerate(all_voices, 1):
            print(f"  [{vi}/{len(all_voices)}] synthesizing voice={voice} ...", flush=True)
            try:
                part, sk = synthesize_voice(
                    voice=voice,
                    texts=texts,
                    pipeline=pipeline,
                    model=model,
                    device=device,
                    samples_dir=samples_dir,
                    out=out,
                    force=args.force,
                )
            except Exception as e:
                print(f"  SKIP voice={voice}: {type(e).__name__}: {e}", flush=True)
                continue
            rows.extend(part)
            skipped += sk
            print(f"  voice={voice} n={len(part)} total={len(rows)} skip={skipped}", flush=True)

    write_jsonl(out / "manifest.jsonl", rows)

    if args.split_mode == "voice":
        val_set = set(val_voices)
        train_rows = [r for r in rows if r["voice"] not in val_set]
        val_rows = [r for r in rows if r["voice"] in val_set]
        if not val_rows:
            raise SystemExit("no val rows — check val voices synthesized OK")
        if not train_rows:
            raise SystemExit("no train rows — check train voices")
    else:
        rng = random.Random(args.seed)
        ids = list(range(len(rows)))
        rng.shuffle(ids)
        n_val = max(1, int(round(len(ids) * args.val_frac))) if len(ids) > 5 else max(1, len(ids) // 5)
        n_val = min(n_val, max(1, len(ids) - 1)) if len(ids) > 1 else 0
        val_idx = set(ids[:n_val])
        train_rows = [rows[i] for i in ids if i not in val_idx]
        val_rows = [rows[i] for i in ids if i in val_idx]

    write_jsonl(out / "train.jsonl", train_rows)
    write_jsonl(out / "val.jsonl", val_rows)

    summary = {
        "n_total": len(rows),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "skipped": skipped,
        "split_mode": args.split_mode,
        "train_voices": sorted({r["voice"] for r in train_rows}),
        "val_voices": sorted({r["voice"] for r in val_rows}),
        "n_texts": len(texts),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"done {summary}", flush=True)


if __name__ == "__main__":
    main()
