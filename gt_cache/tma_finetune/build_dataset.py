"""
Build distillation pairs: Kokoro wav mel + phonemes → duration_proj target.

  .venv\\Scripts\\python.exe -m gt_cache.tma_finetune.build_dataset \\
      --train-voices af_heart,am_michael,af_bella,bm_george \\
      --val-voices af_nicole,am_adam --limit-texts 12 --force
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

from gt_cache.jdc_finetune.common import (  # noqa: E402
    DEFAULT_TEXTS,
    DEFAULT_TRAIN_VOICES,
    DEFAULT_VAL_VOICES,
    list_kokoro_voices_from_cache,
    pick_val_voices,
    tokens_from_phonemes,
    write_jsonl,
)
from gt_cache.tma_finetune import DEFAULT_DATA_DIR  # noqa: E402
from gt_cache.tma_finetune.common import (  # noqa: E402
    phonemes_st2_from_text,
    text_ids_from_phonemes,
)
from gt_cache.utils import wav_to_mel  # noqa: E402
from timbre_encoder.optimize_speaker_ref_s import pack_slot  # noqa: E402


def load_texts_from_csv(
    csv_path: Path,
    max_n: int,
    seed: int,
    min_chars: int = 20,
    max_chars: int = 120,
) -> list[str]:
    import csv

    texts: list[str] = []
    seen: set[str] = set()
    with csv_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
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
                    "n_ph": int(obj["d_kokoro"].numel()),
                    "T": int(obj["mel"].shape[-1]),
                    "status": "cached",
                }
            )
            continue

        phonemes = phonemes_st2_from_text(pipeline, text)
        if not phonemes:
            skipped += 1
            continue
        input_ids = tokens_from_phonemes(model, phonemes)
        if not input_ids:
            skipped += 1
            continue
        text_ids = text_ids_from_phonemes(phonemes)
        if len(text_ids) != len(phonemes):
            skipped += 1
            continue

        ids = torch.LongTensor([[0, *input_ids, 0]]).to(device)
        ref_s = pack_slot(pack, len(phonemes)).to(device)
        with torch.no_grad():
            audio, pred_dur = model.forward_with_tokens(ids, ref_s, speed=1.0)
        pred = pred_dur.detach().float().view(-1).cpu()
        if pred.numel() < 3:
            skipped += 1
            continue
        # core phonemes only (drop BOS/EOS) — matches ST2 text_ids length
        d_kokoro = pred[1:-1].clamp_min(1.0)
        if int(d_kokoro.numel()) != len(phonemes):
            skipped += 1
            continue

        wav = audio.detach().cpu().float().view(-1).numpy()
        mel = wav_to_mel(wav, device=device).detach().float().cpu()
        t_mel = int(mel.shape[-1])
        if t_mel < 16:
            skipped += 1
            continue

        torch.save(
            {
                "mel": mel.contiguous(),
                "d_kokoro": d_kokoro.contiguous(),
                "text_ids": torch.tensor(text_ids, dtype=torch.long),
                "phonemes": phonemes,
                "voice": voice,
                "text": text,
            },
            pt_path,
        )
        rows.append(
            {
                "id": sid,
                "path": str(pt_path.relative_to(out)).replace("\\", "/"),
                "voice": voice,
                "text": text,
                "n_ph": len(phonemes),
                "T": t_mel,
                "status": "ok",
            }
        )
    return rows, skipped


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _trim_balanced(rows: list[dict], n_target: int, seed: int) -> list[dict]:
    """Keep up to n_target rows, round-robin across voices for balance."""
    if n_target <= 0 or len(rows) <= n_target:
        return rows
    by_v: dict[str, list[dict]] = {}
    for r in rows:
        by_v.setdefault(r["voice"], []).append(r)
    for v in by_v:
        rng = random.Random(seed + hash(v) % 10_000)
        rng.shuffle(by_v[v])
    voices = sorted(by_v.keys())
    out: list[dict] = []
    idx = {v: 0 for v in voices}
    while len(out) < n_target:
        progressed = False
        for v in voices:
            i = idx[v]
            if i < len(by_v[v]):
                out.append(by_v[v][i])
                idx[v] = i + 1
                progressed = True
                if len(out) >= n_target:
                    break
        if not progressed:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build TMA<-Kokoro duration dataset")
    ap.add_argument("--out", type=str, default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--all-voices", action="store_true")
    ap.add_argument("--n-val-voices", type=int, default=4)
    ap.add_argument(
        "--train-voices",
        type=str,
        default="af_heart,am_michael,af_bella,bm_george,af_sarah,am_fenrir",
    )
    ap.add_argument(
        "--val-voices",
        type=str,
        default="af_nicole,am_adam,bf_emma,bm_lewis",
    )
    ap.add_argument("--texts", nargs="*", default=None)
    ap.add_argument("--texts-csv", type=str, default="")
    ap.add_argument("--limit-texts", type=int, default=12)
    ap.add_argument(
        "--train-clips",
        type=int,
        default=0,
        help="Target train clip count (0=keep all train-voice synths)",
    )
    ap.add_argument(
        "--val-clips",
        type=int,
        default=0,
        help="Target val/test clip count (0=keep all val-voice synths)",
    )
    ap.add_argument("--min-chars", type=int, default=20)
    ap.add_argument("--max-chars", type=int, default=120)
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
        seen: set[str] = set()
        all_voices = []
        for v in train_voices + val_voices:
            if v not in seen:
                seen.add(v)
                all_voices.append(v)

    overlap = set(train_voices) & set(val_voices)
    if overlap:
        raise SystemExit(f"train/val voice overlap: {sorted(overlap)}")

    # How many texts each split needs to reach clip targets
    n_texts_train = (
        _ceil_div(args.train_clips, max(1, len(train_voices)))
        if args.train_clips > 0
        else args.limit_texts
    )
    n_texts_val = (
        _ceil_div(args.val_clips, max(1, len(val_voices)))
        if args.val_clips > 0
        else args.limit_texts
    )
    n_texts_need = max(n_texts_train, n_texts_val, args.limit_texts)

    if args.texts:
        pool = list(args.texts)
    else:
        csv_path = Path(args.texts_csv) if args.texts_csv else (
            ROOT / "timbre_encoder" / "manifests" / "train-clean-100.filtered.csv"
        )
        if not csv_path.is_file():
            csv_path = ROOT / "timbre_encoder" / "manifests" / "dev-clean.filtered.csv"
        if csv_path.is_file():
            pool = load_texts_from_csv(
                csv_path,
                n_texts_need,
                args.seed,
                min_chars=args.min_chars,
                max_chars=args.max_chars,
            )
        else:
            pool = list(DEFAULT_TEXTS)

    if not pool:
        raise SystemExit("no texts")
    if len(pool) < n_texts_need:
        print(
            f"WARNING: only {len(pool)} texts available, need {n_texts_need}",
            flush=True,
        )

    texts_train = pool[: min(n_texts_train, len(pool))]
    texts_val = pool[: min(n_texts_val, len(pool))]

    print(
        f"device={device} train_v={len(train_voices)} val_v={len(val_voices)} "
        f"texts_train={len(texts_train)} texts_val={len(texts_val)} "
        f"target_train={args.train_clips or 'all'} target_val={args.val_clips or 'all'} "
        f"out={out}",
        flush=True,
    )
    print(f"  train_voices={train_voices}", flush=True)
    print(f"  val_voices={val_voices}", flush=True)

    pipeline = KPipeline(lang_code="a", model=False)
    model = KModel(repo_id="hexgrad/Kokoro-82M").to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    rows: list[dict] = []
    skipped = 0
    val_set = set(val_voices)
    for vi, voice in enumerate(all_voices, 1):
        texts = texts_val if voice in val_set else texts_train
        print(
            f"  [{vi}/{len(all_voices)}] voice={voice} texts={len(texts)}",
            flush=True,
        )
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
        print(f"    n={len(part)} total={len(rows)} skip={skipped}", flush=True)

    write_jsonl(out / "manifest.jsonl", rows)
    train_rows = [r for r in rows if r["voice"] not in val_set]
    val_rows = [r for r in rows if r["voice"] in val_set]
    if args.train_clips > 0:
        train_rows = _trim_balanced(train_rows, args.train_clips, args.seed)
    if args.val_clips > 0:
        val_rows = _trim_balanced(val_rows, args.val_clips, args.seed + 1)
    if not train_rows or not val_rows:
        raise SystemExit(f"bad split train={len(train_rows)} val={len(val_rows)}")
    write_jsonl(out / "train.jsonl", train_rows)
    write_jsonl(out / "val.jsonl", val_rows)
    summary = {
        "n_total": len(train_rows) + len(val_rows),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "n_manifest": len(rows),
        "skipped": skipped,
        "train_voices": sorted({r["voice"] for r in train_rows}),
        "val_voices": sorted({r["voice"] for r in val_rows}),
        "n_texts_train": len(texts_train),
        "n_texts_val": len(texts_val),
        "train_clips_target": args.train_clips,
        "val_clips_target": args.val_clips,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"done {summary}", flush=True)


if __name__ == "__main__":
    main()
