"""
Listen: LibriTTS ref → E_a → mapper → Kokoro pack vs original / ST2 recon.

  .venv\\Scripts\\python.exe -m timbre_encoder.make_listen_mapper
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kokoro.pipeline import KPipeline

from timbre_encoder.dual_encoder import crop_mel_for_style
from timbre_encoder.librispeech_dataset import text_to_kokoro_phonemes
from timbre_encoder.mel_preprocess import SAMPLE_RATE, waveform_to_mel
from timbre_encoder.st2_kokoro_mapper import St2ToKokoroMapper
from timbre_encoder.st2_runtime import (
    decode_with_style,
    load_frozen_st2,
    phonemes_to_st2_ids,
    prepare_asr_f0_n,
)
from timbre_encoder.train_encoders_styletts2 import AbsoluteDualStyleEncoder
from timbre_encoder.voice_pack import assemble_voice_pack


def synth_kokoro(pipeline: KPipeline, text: str, pack: torch.Tensor) -> np.ndarray:
    chunks = []
    for r in pipeline(text, voice=pack, speed=1):
        if r.audio is not None:
            chunks.append(r.audio.detach().cpu().numpy())
    return np.concatenate(chunks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ea-ckpt",
        default=str(ROOT / "timbre_encoder" / "runs" / "st2_encoders_v2" / "encoders_final.pt"),
    )
    ap.add_argument(
        "--mapper-ckpt",
        default=str(ROOT / "timbre_encoder" / "runs" / "st2_kokoro_mapper_v2" / "mapper_final.pt"),
    )
    ap.add_argument(
        "--out",
        default=str(ROOT / "timbre_encoder" / "runs" / "st2_kokoro_mapper_v2" / "listen_test"),
    )
    ap.add_argument(
        "--manifest",
        default=str(ROOT / "timbre_encoder" / "manifests" / "dev-clean.filtered.csv"),
    )
    ap.add_argument("--baseline", default="af_heart", help="Prosody + residual anchor")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st2, _ = load_frozen_st2(device)
    ea = AbsoluteDualStyleEncoder.from_st2(st2).to(device)
    ea.load_state_dict(torch.load(args.ea_ckpt, map_location=device, weights_only=False)["encoders"])
    ea.eval()

    mstate = torch.load(args.mapper_ckpt, map_location=device, weights_only=False)
    mapper = St2ToKokoroMapper(
        residual_scale=mstate.get("residual_scale", 0.35),
        mode=mstate.get("mode", "residual"),
        hidden=mstate.get("hidden", 256),
    ).to(device)
    mapper.load_state_dict(mstate["mapper"])
    mapper.eval()

    pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    baseline_pack = pipe.load_voice(args.baseline)
    baseline_timbre = baseline_pack[100, 0, :128].float().to(device)

    rows = list(csv.DictReader(open(args.manifest, encoding="utf-8")))
    html = [
        "<!DOCTYPE html><html><body><h1>ST2→Kokoro mapper listen</h1>",
        f"<p>E_a={args.ea_ckpt}<br>mapper={args.mapper_ckpt}<br>"
        f"mode={mstate.get('mode')} scale={mstate.get('residual_scale')}<br>"
        f"baseline={args.baseline}</p>",
    ]

    with torch.no_grad():
        for i, row in enumerate(rows[: args.n]):
            text = row["text"]
            wave, _ = librosa.load(row["audio_path"], sr=SAMPLE_RATE, mono=True)
            wave, _ = librosa.effects.trim(wave, top_db=30)
            wave = wave[: int(4 * SAMPLE_RATE)]
            mel = waveform_to_mel(torch.from_numpy(wave), device=device)
            mel_crop = crop_mel_for_style(mel, max_frames=192, min_frames=64)

            s_a, _ = ea(mel_crop)
            timbre = mapper(
                s_a, baseline_timbre if mstate.get("mode", "residual") == "residual" else None
            ).squeeze(0).cpu()
            pack = assemble_voice_pack(timbre, baseline_pack)
            k_wav = synth_kokoro(pipe, text, pack)

            # ST2 recon reference (same E_a, no Kokoro)
            phonemes = text_to_kokoro_phonemes(text, pipe)
            text_ids = phonemes_to_st2_ids(phonemes, device)
            asr, F0, Ncurve, mel_even = prepare_asr_f0_n(st2, mel.squeeze(0), text_ids)
            y_st2 = decode_with_style(st2, asr, F0, Ncurve, s_a).detach().cpu().numpy().squeeze()

            # baseline-only Kokoro (no mapper delta)
            pack_base = assemble_voice_pack(baseline_timbre.cpu(), baseline_pack)
            k_base = synth_kokoro(pipe, text, pack_base)

            stem = f"{i+1:02d}"
            sf.write(out / f"{stem}_original.wav", wave, SAMPLE_RATE)
            sf.write(out / f"{stem}_st2_Ea.wav", y_st2, SAMPLE_RATE)
            sf.write(out / f"{stem}_kokoro_mapped.wav", k_wav, SAMPLE_RATE)
            sf.write(out / f"{stem}_kokoro_baseline_only.wav", k_base, SAMPLE_RATE)
            html += [
                f"<h3>{stem} {text[:80]}</h3>",
                f"<p>original</p><audio controls src='{stem}_original.wav'></audio>",
                f"<p>ST2 + E_a</p><audio controls src='{stem}_st2_Ea.wav'></audio>",
                f"<p>Kokoro mapped</p><audio controls src='{stem}_kokoro_mapped.wav'></audio>",
                f"<p>Kokoro baseline only</p><audio controls src='{stem}_kokoro_baseline_only.wav'></audio>",
            ]
            print(
                f"{stem} s_a_norm={float(s_a.norm()):.3f} timbre_norm={float(timbre.norm()):.3f} "
                f"l1_vs_base={float((timbre - baseline_timbre.cpu()).abs().mean()):.4f}",
                flush=True,
            )

    html.append("</body></html>")
    (out / "index.html").write_text("\n".join(html), encoding="utf-8")
    print(f"DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
