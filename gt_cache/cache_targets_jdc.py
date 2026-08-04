"""Fill kokoro_targets_v1 NPZs with full-utterance JDC F0 + energy (no CTC).

Skips files that already contain d_gt (complete RunPod/CTC caches).
Writes partial NPZs with stage='jdc' for later CTC duration fill.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gt_cache.cache_runpod_targets import (  # noqa: E402
    HOP_LENGTH,
    MEL_MEAN,
    MEL_STD,
    SAMPLE_RATE,
    load_jdc,
    load_phonemes_map,
)

STAGE = "jdc"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["train-clean-100", "dev-clean"])
    ap.add_argument("--kokoro-config", type=Path, required=True)
    ap.add_argument("--styletts2-root", type=Path, required=True)
    ap.add_argument("--jdc-weights", type=Path, required=True)
    ap.add_argument("--phonemes-map", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    return ap.parse_args()


def load_wav24(path: Path) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        wav = soxr.resample(wav, sr, SAMPLE_RATE).astype(np.float32)
    return wav


def already_done(path: Path, force: bool) -> bool:
    if force or not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=True) as z:
            files = set(z.files)
            if "d_gt" in files and "f0" in files:
                return True
            if "f0" in files and "n" in files and str(z["stage"]) == STAGE:
                return True
    except Exception:
        return False
    return False


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = json.loads(args.kokoro_config.read_text(encoding="utf-8"))
    kokoro_vocab: dict[str, int] = config["vocab"]
    phonemes_map = load_phonemes_map(args.phonemes_map)
    print(f"loaded phonemes_map={len(phonemes_map)}", flush=True)

    jdc = load_jdc(args.styletts2_root, args.jdc_weights, device)
    mel_fn = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=2048,
        win_length=1200,
        hop_length=HOP_LENGTH,
        n_mels=80,
    ).to(device)

    wav_paths: list[Path] = []
    for split in args.splits:
        wav_paths.extend(sorted((args.data_root / split).rglob("*.wav")))
    keep = []
    for p in wav_paths:
        key = str(p.relative_to(args.data_root)).replace("\\", "/")
        if key in phonemes_map:
            keep.append(p)
    print(f"filtered by phonemes_map: {len(keep)}/{len(wav_paths)}", flush=True)
    wav_paths = keep
    if args.limit > 0:
        wav_paths = wav_paths[: args.limit]
    if args.num_shards > 1:
        wav_paths = wav_paths[args.shard_index :: args.num_shards]

    pending = [
        p
        for p in wav_paths
        if not already_done(
            args.output_root / p.relative_to(args.data_root).with_suffix(".npz"),
            args.force,
        )
    ]
    print(
        f"device={device} shard={args.shard_index}/{args.num_shards} "
        f"pending={len(pending)}/{len(wav_paths)} force={args.force}",
        flush=True,
    )

    done = failed = 0
    for i, path in enumerate(pending, 1):
        rel = path.relative_to(args.data_root)
        out = args.output_root / rel.with_suffix(".npz")
        key = str(rel).replace("\\", "/")
        try:
            cached = phonemes_map[key]
            phonemes = cached["phonemes"]
            text = cached.get("text") or ""
            tokens = [c for c in phonemes if c in kokoro_vocab and c != "$"]
            if not tokens:
                raise ValueError("no Kokoro tokens")
            ids = np.asarray([kokoro_vocab[x] for x in tokens], dtype=np.uint16)

            wav24 = load_wav24(path)
            wave = torch.from_numpy(wav24).to(device)
            mel = mel_fn(wave)
            mel = (torch.log(mel.clamp_min(1e-5)) - MEL_MEAN) / MEL_STD
            with torch.inference_mode():
                f0, _, _ = jdc(mel.unsqueeze(0).unsqueeze(0))
            f0 = f0.float().squeeze().reshape(-1)
            n_curve = torch.log(
                torch.exp(mel * MEL_STD + MEL_MEAN).norm(dim=0).clamp_min(1e-8)
            )
            mel_len = int(mel.shape[-1])
            if int(f0.numel()) != mel_len:
                raise ValueError(f"JDC length {f0.numel()} != mel {mel_len}")

            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out,
                f0=f0.cpu().numpy().astype(np.float32),
                n=n_curve.cpu().numpy().astype(np.float32),
                phoneme_ids=ids,
                phonemes=np.asarray(phonemes),
                text=np.asarray(text),
                mel_len=np.asarray(mel_len, dtype=np.int32),
                stage=np.asarray(STAGE),
            )
            done += 1
        except Exception as exc:
            failed += 1
            print(f"JDC_FAIL {path}: {type(exc).__name__}: {exc}", flush=True)
        if i % args.log_every == 0 or i == len(pending):
            print(f"[{i}/{len(pending)}] cached={done} failed={failed}", flush=True)

    print(f"DONE stage=jdc cached={done} failed={failed} root={args.output_root}", flush=True)


if __name__ == "__main__":
    main()
