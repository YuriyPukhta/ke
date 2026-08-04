"""Compare Kokoro-native, Wav2Vec2 CTC, and StyleTTS2 TMA alignments.

The experiment synthesizes one phrase with Kokoro while retaining its native
duration projection, then aligns that exact generated waveform with CTC and
TMA. Results are normalized onto the same 24 kHz / hop-300 mel timeline.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

from kokoro.pipeline import KPipeline  # noqa: E402
from kokoro_revers.model import KModel  # noqa: E402

from gt_cache.cache_runpod_targets import (  # noqa: E402
    PHONE_MAP,
    allocate_with_min,
    durations_from_spans,
    make_groups,
)
from timbre_encoder.aligner import FrozenTextAligner  # noqa: E402
from timbre_encoder.kokoro_embed_train.common import freeze_module, pack_slot, to_1d  # noqa: E402
from timbre_encoder.mel_preprocess import HOP_LENGTH, SAMPLE_RATE, waveform_to_mel  # noqa: E402
from timbre_encoder.phoneme_norm import normalize_kokoro_phonemes_for_st2  # noqa: E402


def normalize_to_total(durations: list[int], total: int) -> list[int]:
    raw = np.asarray(durations, dtype=np.float64)
    minimum = np.ones(len(raw), dtype=np.int64)
    return allocate_with_min(raw, minimum, total).astype(int).tolist()


def metrics(reference: list[int], estimate: list[int]) -> dict[str, float]:
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(estimate, dtype=np.float64)
    ba = np.cumsum(a)[:-1]
    bb = np.cumsum(b)[:-1]
    return {
        "duration_mae_frames": float(np.mean(np.abs(a - b))),
        "boundary_mae_frames": float(np.mean(np.abs(ba - bb))),
        "boundary_p95_frames": float(np.quantile(np.abs(ba - bb), 0.95)),
        "duration_corr": float(np.corrcoef(a, b)[0, 1]),
    }


def ctc_align(
    wav: np.ndarray,
    phonemes: str,
    mel_len: int,
    model_root: Path,
    phonealign_root: Path,
    device: torch.device,
) -> tuple[list[int], float]:
    sys.path.insert(0, str(phonealign_root / "src"))
    from phonalign.acoustic import AcousticModel
    from phonalign.align import PhoneVocabMapper, ctc_forced_align

    acoustic = AcousticModel(model_id=str(model_root), device=str(device))
    mapper = PhoneVocabMapper(acoustic.vocab, overrides=PHONE_MAP)
    tokens = list(phonemes)
    groups, labels = make_groups(tokens)
    token_ids: list[int] = []
    tokens_per_group: list[int] = []
    for label in labels:
        ids = mapper.map_phone(label, word="?", language="en-us")
        token_ids.extend(ids)
        tokens_per_group.append(len(ids))

    wav16 = soxr.resample(wav, SAMPLE_RATE, 16_000).astype(np.float32)
    log_probs = acoustic.batch_emissions([wav16])[0]
    spans = ctc_forced_align(
        log_probs,
        torch.tensor(token_ids, dtype=torch.long),
        acoustic.blank_id,
    )
    d_gt, score = durations_from_spans(
        spans,
        tokens_per_group,
        groups,
        len(tokens),
        log_probs.shape[0],
        mel_len,
    )
    return d_gt.astype(int).tolist(), score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="The storm had become shapeless.")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument(
        "--ctc-model",
        type=Path,
        default=ROOT / "models/wav2vec2-lv60-espeak",
    )
    ap.add_argument(
        "--phonealign-root",
        type=Path,
        default=ROOT / "models/phonealign-en-lao",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "gt_cache/results/kokoro_ctc_tma",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    _, phonemes_raw, _ = next(pipeline(args.text, voice=args.voice, speed=1.0))
    phonemes = normalize_kokoro_phonemes_for_st2(phonemes_raw).normalized

    model = KModel(repo_id="hexgrad/Kokoro-82M").to(device).eval()
    freeze_module(model)
    pack = pipeline.load_voice(args.voice).to(device)
    ref_s = pack_slot(pack, len(phonemes))
    if ref_s.dim() == 1:
        ref_s = ref_s.unsqueeze(0)
    ids = [model.vocab[p] for p in phonemes if p in model.vocab]
    if len(ids) != len(phonemes):
        raise RuntimeError(f"Kokoro vocab drop: {len(ids)} != {len(phonemes)}")
    input_ids = torch.LongTensor([[0, *ids, 0]]).to(device)
    with torch.no_grad():
        audio, pred_dur = model.forward_with_tokens(input_ids, ref_s, speed=1.0)
    wav = audio.detach().float().cpu().numpy().astype(np.float32)
    pred_full = [int(x) for x in to_1d(pred_dur).cpu().tolist()]
    d_kokoro_raw = pred_full[1:-1]
    # The generated waveform includes Kokoro's BOS/EOS frames. CTC and TMA,
    # which only see core phonemes, assign those edge frames to the first/last
    # phone. Merge them likewise before comparing boundaries.
    d_kokoro_raw[0] += pred_full[0]
    d_kokoro_raw[-1] += pred_full[-1]
    del model, pack, ref_s, input_ids, audio, pred_dur
    torch.cuda.empty_cache()

    mel = waveform_to_mel(torch.from_numpy(wav), device=device).squeeze(0)
    mel_len = int(mel.shape[-1])

    aligner = FrozenTextAligner(device=device)
    _, d_tma_down, _ = aligner.align(mel, phonemes)
    d_tma_raw = [int(x) for x in d_tma_down.cpu().tolist()]
    comparison_len = sum(d_tma_raw)
    d_kokoro = normalize_to_total(d_kokoro_raw, comparison_len)
    d_tma = normalize_to_total(d_tma_raw, comparison_len)
    del aligner, mel
    torch.cuda.empty_cache()

    d_ctc_mel, align_score = ctc_align(
        wav,
        phonemes,
        mel_len,
        args.ctc_model,
        args.phonealign_root,
        device,
    )
    d_ctc = normalize_to_total(d_ctc_mel, comparison_len)

    result = {
        "text": args.text,
        "voice": args.voice,
        "phonemes_raw": phonemes_raw,
        "phonemes": phonemes,
        "frame_ms": 1000 * HOP_LENGTH / SAMPLE_RATE * 2,
        "mel_frames": mel_len,
        "comparison_frames": comparison_len,
        "raw_sums": {
            "kokoro": sum(d_kokoro_raw),
            "ctc_mel": sum(d_ctc_mel),
            "tma_downsampled": sum(d_tma_raw),
        },
        "durations": {
            "kokoro_native": d_kokoro,
            "wav2vec2_ctc": d_ctc,
            "styletts2_tma": d_tma,
        },
        "metrics_vs_kokoro": {
            "wav2vec2_ctc": metrics(d_kokoro, d_ctc),
            "styletts2_tma": metrics(d_kokoro, d_tma),
        },
        "ctc_align_score": align_score,
        "notes": [
            "All three duration arrays are projected onto the same mel timeline.",
            "TMA internally aligns at half mel resolution (n_down=1), then is rescaled.",
            "Kokoro native duration is the duration_proj actually used to synthesize this waveform.",
        ],
    }
    sf.write(args.out / "sample.wav", wav, SAMPLE_RATE)
    (args.out / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
