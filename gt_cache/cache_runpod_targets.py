"""Cache Kokoro/Misaki phoneme durations, JDC F0, and energy for LibriTTS-R.

This standalone RunPod-oriented preprocessor uses:
  * Misaki for the exact Kokoro English phoneme sequence;
  * facebook/wav2vec2-lv-60-espeak-cv-ft + CTC Viterbi for alignment;
  * StyleTTS2 JDCNet for frame-level F0;
  * StyleTTS2 log_norm for frame-level energy.

Each utterance is written as one compressed NPZ.  Features are computed on the
whole utterance; training code should later choose phoneme-aligned crops.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch
import torchaudio


SAMPLE_RATE = 24_000
HOP_LENGTH = 300
MEL_MEAN = -4.0
MEL_STD = 4.0

# Misaki EN_PHONES.md (+ Kokoro ɾ→T) -> eSpeak IPA CTC tokens.
PHONE_MAP: dict[str, list[str]] = {
    "A": ["eɪ"],
    "I": ["aɪ"],
    "W": ["aʊ"],
    "Y": ["ɔɪ"],
    "O": ["oʊ"],
    "Q": ["əʊ"],
    "ʤ": ["dʒ"],
    "ʧ": ["tʃ"],
    "ᵊ": ["ə"],
    "T": ["ɾ"],  # Kokoro stores American flap as T (pipeline: ɾ => T)
}
STRESSES = {"ˈ", "ˌ"}
MISAKI_SEGMENTS = set(
    "AIWYbdfhijklmnpstuvwzðŋɑɔəɛɜɡɪɹʃʊʌʒʤʧθᵊOæɾᵻQaɒːT"
)


@dataclass
class Item:
    wav_path: Path
    rel_wav: Path
    text: str
    phonemes: str
    tokens: list[str]
    groups: list[list[int]]
    group_labels: list[str]
    token_ids: list[int]
    tokens_per_group: list[int]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["train-clean-100", "dev-clean"])
    ap.add_argument("--kokoro-config", type=Path, required=True)
    ap.add_argument("--ctc-model", type=Path, required=True)
    ap.add_argument("--phonealign-root", type=Path, required=True)
    ap.add_argument("--styletts2-root", type=Path, required=True)
    ap.add_argument("--jdc-weights", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0, help="0-based worker shard")
    ap.add_argument(
        "--phonemes-map",
        type=Path,
        default=None,
        help="JSONL with {audio,phonemes[,text]} keyed by path relative to data-root",
    )
    ap.add_argument(
        "--no-g2p",
        action="store_true",
        help="Never call Misaki; skip utterances missing from --phonemes-map",
    )
    return ap.parse_args()


def transcript_for(wav_path: Path) -> str:
    for suffix in (".normalized.txt", ".original.txt"):
        p = wav_path.with_suffix(suffix)
        if p.is_file():
            return p.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(f"no transcript beside {wav_path}")


def load_phonemes_map(path: Path) -> dict[str, dict[str, str]]:
    """Map relative audio path -> {phonemes, text?}."""
    out: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = str(row["audio"]).replace("\\", "/")
            out[key] = {
                "phonemes": str(row["phonemes"]),
                "text": str(row.get("text") or ""),
            }
    return out


def is_delimiter(token: str) -> bool:
    return (
        token.isspace()
        or token not in MISAKI_SEGMENTS | STRESSES
        or unicodedata.category(token).startswith("P")
    )


def make_groups(tokens: list[str]) -> tuple[list[list[int]], list[str]]:
    """Partition tokens into acoustic groups accepted by the phoneme CTC model.

    Stress and delimiters are attached to the following phone, while a length
    mark is attached to the preceding vowel.  Thus every Kokoro token receives
    a duration even though the CTC vocabulary has no punctuation/space token.
    """
    groups: list[list[int]] = []
    labels: list[str] = []
    pending: list[int] = []
    for i, tok in enumerate(tokens):
        if tok in STRESSES or is_delimiter(tok):
            pending.append(i)
            continue
        if tok == "ː" and groups:
            groups[-1].append(i)
            labels[-1] += tok
            continue
        groups.append([*pending, i])
        labels.append(tok)
        pending.clear()
    if pending:
        if not groups:
            raise ValueError("phoneme sequence contains no acoustic phones")
        groups[-1].extend(pending)
    return groups, labels


def prepare_item(
    wav_path: Path,
    data_root: Path,
    g2p,
    kokoro_vocab: dict[str, int],
    mapper,
    phonemes_map: dict[str, dict[str, str]] | None = None,
    no_g2p: bool = False,
) -> Item:
    rel_wav = wav_path.relative_to(data_root)
    rel_key = str(rel_wav).replace("\\", "/")
    cached = phonemes_map.get(rel_key) if phonemes_map else None
    if cached and cached.get("phonemes"):
        phonemes = cached["phonemes"]
        text = cached.get("text") or transcript_for(wav_path)
    elif no_g2p:
        raise FileNotFoundError(f"no cached Misaki phonemes for {rel_key}")
    else:
        text = transcript_for(wav_path)
        if g2p is None:
            raise RuntimeError("Misaki G2P unavailable and no phonemes-map entry")
        phonemes, _ = g2p(text)
    if not phonemes:
        raise ValueError("empty phonemes")
    tokens = [c for c in phonemes if c in kokoro_vocab and c != "$"]
    if not tokens:
        raise ValueError("no Kokoro tokens")
    groups, labels = make_groups(tokens)

    token_ids: list[int] = []
    tokens_per_group: list[int] = []
    for label in labels:
        ids = mapper.map_phone(label, word="?", language="en-us")
        token_ids.extend(ids)
        tokens_per_group.append(len(ids))

    return Item(
        wav_path=wav_path,
        rel_wav=rel_wav,
        text=text,
        phonemes=phonemes,
        tokens=tokens,
        groups=groups,
        group_labels=labels,
        token_ids=token_ids,
        tokens_per_group=tokens_per_group,
    )


def load_audio(path: Path) -> tuple[np.ndarray, np.ndarray]:
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    wav24 = soxr.resample(wav, sr, SAMPLE_RATE).astype(np.float32) if sr != SAMPLE_RATE else wav
    wav16 = soxr.resample(wav24, SAMPLE_RATE, 16_000).astype(np.float32)
    return wav24, wav16


def allocate_with_min(raw: np.ndarray, minimum: np.ndarray, total: int) -> np.ndarray:
    """Integer allocation summing to total while respecting per-unit minima."""
    if total < int(minimum.sum()):
        raise ValueError(f"mel too short: {total} frames for {minimum.sum()} tokens")
    out = minimum.astype(np.int64).copy()
    remain = total - int(out.sum())
    if remain == 0:
        return out
    weight = np.maximum(raw.astype(np.float64) - minimum, 0.0)
    if not np.isfinite(weight).all() or weight.sum() <= 0:
        weight = np.ones_like(weight)
    exact = weight / weight.sum() * remain
    add = np.floor(exact).astype(np.int64)
    out += add
    left = total - int(out.sum())
    if left:
        order = np.argsort(-(exact - add))
        out[order[:left]] += 1
    return out


def durations_from_spans(
    spans: list[tuple[int, int, int, float]],
    tokens_per_group: list[int],
    groups: list[list[int]],
    n_tokens: int,
    n_ctc_frames: int,
    n_mel_frames: int,
) -> tuple[np.ndarray, float]:
    """Merge CTC-token spans, allocate blanks, then expand groups to Kokoro tokens."""
    merged: list[tuple[int, int, float]] = []
    cursor = 0
    for n in tokens_per_group:
        chunk = spans[cursor : cursor + n]
        cursor += n
        merged.append((chunk[0][1], chunk[-1][2], float(np.mean([x[3] for x in chunk]))))

    bounds = np.zeros(len(merged) + 1, dtype=np.float64)
    bounds[-1] = n_ctc_frames
    for i in range(1, len(merged)):
        bounds[i] = (merged[i - 1][1] + merged[i][0]) / 2.0
    raw_group = np.diff(bounds) * (n_mel_frames / n_ctc_frames)
    minimum = np.asarray([len(g) for g in groups], dtype=np.int64)
    group_dur = allocate_with_min(raw_group, minimum, n_mel_frames)

    d_gt = np.ones(n_tokens, dtype=np.int64)
    for g, dur in zip(groups, group_dur):
        d_gt[g[-1]] += int(dur) - len(g)
    if int(d_gt.sum()) != n_mel_frames or np.any(d_gt < 1):
        raise AssertionError("invalid duration allocation")
    return d_gt.astype(np.uint16), float(np.mean([x[2] for x in merged]))


def load_jdc(styletts2_root: Path, weights: Path, device: torch.device):
    sys.path.insert(0, str(styletts2_root))
    from Utils.JDC.model import JDCNet

    model = JDCNet(num_class=1, seq_len=192)
    state = torch.load(weights, map_location="cpu", weights_only=False)["net"]
    model.load_state_dict(state)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sys.path.insert(0, str(args.phonealign_root / "src"))
    from phonalign.acoustic import AcousticModel
    from phonalign.align import PhoneVocabMapper, ctc_forced_align

    config = json.loads(args.kokoro_config.read_text(encoding="utf-8"))
    kokoro_vocab: dict[str, int] = config["vocab"]
    phonemes_map: dict[str, dict[str, str]] | None = None
    if args.phonemes_map is not None:
        phonemes_map = load_phonemes_map(args.phonemes_map)
        print(f"loaded phonemes_map={len(phonemes_map)} from {args.phonemes_map}", flush=True)
    if args.no_g2p and not phonemes_map:
        raise SystemExit("--no-g2p requires --phonemes-map")

    g2p = None
    if not args.no_g2p:
        from misaki import en

        g2p = en.G2P(trf=False, british=False, fallback=None)

    acoustic = AcousticModel(model_id=str(args.ctc_model), device=str(device))
    mapper = PhoneVocabMapper(acoustic.vocab, overrides=PHONE_MAP)
    jdc = load_jdc(args.styletts2_root, args.jdc_weights, device)
    mel_fn = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=2048,
        win_length=1200,
        hop_length=HOP_LENGTH,
        n_mels=80,
    ).to(device)

    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit("--shard-index must be in [0, num-shards)")

    wav_paths: list[Path] = []
    for split in args.splits:
        wav_paths.extend(sorted((args.data_root / split).rglob("*.wav")))
    if phonemes_map is not None and args.no_g2p:
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
    print(
        f"device={device} shard={args.shard_index}/{args.num_shards} "
        f"utterances={len(wav_paths)} batch={args.batch_size} no_g2p={args.no_g2p}",
        flush=True,
    )

    manifest_name = (
        "manifest.jsonl"
        if args.num_shards == 1
        else f"manifest.shard{args.shard_index}.jsonl"
    )
    manifest_path = args.output_root / manifest_name
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    failed = 0

    for batch_start in range(0, len(wav_paths), args.batch_size):
        paths = wav_paths[batch_start : batch_start + args.batch_size]
        prepared: list[Item] = []
        wav24s: list[np.ndarray] = []
        wav16s: list[np.ndarray] = []
        for path in paths:
            out = args.output_root / path.relative_to(args.data_root).with_suffix(".npz")
            if not args.force and out.is_file():
                try:
                    with np.load(out, allow_pickle=True) as z:
                        if "d_gt" in z.files and "f0" in z.files:
                            continue  # complete cache
                except Exception:
                    pass
            try:
                item = prepare_item(
                    path,
                    args.data_root,
                    g2p,
                    kokoro_vocab,
                    mapper,
                    phonemes_map=phonemes_map,
                    no_g2p=args.no_g2p,
                )
                wav24, wav16 = load_audio(path)
                prepared.append(item)
                wav24s.append(wav24)
                wav16s.append(wav16)
            except Exception as exc:
                failed += 1
                print(f"PREP_FAIL {path}: {type(exc).__name__}: {exc}", flush=True)
        if not prepared:
            continue

        emissions = acoustic.batch_emissions(wav16s)
        for item, wav24, log_probs in zip(prepared, wav24s, emissions):
            out = args.output_root / item.rel_wav.with_suffix(".npz")
            try:
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
                if f0.numel() != mel_len:
                    raise ValueError(f"JDC length {f0.numel()} != mel {mel_len}")

                spans = ctc_forced_align(
                    log_probs,
                    torch.tensor(item.token_ids, dtype=torch.long),
                    acoustic.blank_id,
                )
                d_gt, align_score = durations_from_spans(
                    spans,
                    item.tokens_per_group,
                    item.groups,
                    len(item.tokens),
                    log_probs.shape[0],
                    mel_len,
                )
                ids = np.asarray([kokoro_vocab[x] for x in item.tokens], dtype=np.uint16)
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    out,
                    f0=f0.cpu().numpy().astype(np.float32),
                    n=n_curve.cpu().numpy().astype(np.float32),
                    d_gt=d_gt,
                    phoneme_ids=ids,
                    phonemes=np.asarray(item.phonemes),
                    text=np.asarray(item.text),
                    mel_len=np.asarray(mel_len, dtype=np.int32),
                    align_score=np.asarray(align_score, dtype=np.float32),
                )
                record = {
                    "audio": str(item.rel_wav),
                    "cache": str(out.relative_to(args.output_root)),
                    "mel_len": mel_len,
                    "phoneme_len": len(ids),
                    "align_score": align_score,
                    "speaker_id": item.rel_wav.parts[-3],
                }
                with manifest_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                done += 1
            except Exception as exc:
                failed += 1
                print(f"CACHE_FAIL {item.wav_path}: {type(exc).__name__}: {exc}", flush=True)
        seen = batch_start + len(paths)
        if done % args.log_every < len(prepared) or failed:
            print(f"[{seen}/{len(wav_paths)}] cached={done} failed={failed}", flush=True)

    print(f"DONE cached={done} failed={failed} root={args.output_root}", flush=True)


if __name__ == "__main__":
    main()
