"""Listen-test: ST2 reconstruct with trained E_a vs official ST2 style.

  .venv\\Scripts\\python.exe -m timbre_encoder.make_listen_st2_encoders \\
      --ckpt timbre_encoder/runs/st2_encoders_v2/encoders_final.pt \\
      --out  timbre_encoder/runs/st2_encoders_v2/listen_test
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import librosa
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kokoro.pipeline import KPipeline

from timbre_encoder.dual_encoder import crop_mel_for_style
from timbre_encoder.librispeech_dataset import text_to_kokoro_phonemes
from timbre_encoder.mel_preprocess import SAMPLE_RATE, waveform_to_mel
from timbre_encoder.st2_runtime import (
    decode_with_style,
    load_frozen_st2,
    phonemes_to_st2_ids,
    prepare_asr_f0_n,
)
from timbre_encoder.train_encoders_styletts2 import AbsoluteDualStyleEncoder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        type=str,
        default=str(ROOT / "timbre_encoder" / "runs" / "st2_encoders_v2" / "encoders_final.pt"),
    )
    ap.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "timbre_encoder" / "runs" / "st2_encoders_v2" / "listen_test"),
    )
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument(
        "--manifest",
        type=str,
        default=str(ROOT / "timbre_encoder" / "manifests" / "dev-clean.filtered.csv"),
    )
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st2, _ = load_frozen_st2(device)
    enc = AbsoluteDualStyleEncoder.from_st2(st2).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    enc.load_state_dict(state["encoders"])
    enc.eval()

    pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    rows = list(csv.DictReader(open(args.manifest, encoding="utf-8")))
    html = ["<!DOCTYPE html><html><body><h1>ST2 encoders listen</h1>", f"<p>{args.ckpt}</p>"]

    with torch.no_grad():
        for i, row in enumerate(rows[: args.n * 20]):
            if i >= args.n:
                break
            wav_path = Path(row["audio_path"])
            text = row["text"]
            wave, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
            wave, _ = librosa.effects.trim(wave, top_db=30)
            wave = wave[: int(4 * SAMPLE_RATE)]
            mel = waveform_to_mel(torch.from_numpy(wave), device=device)
            phonemes = text_to_kokoro_phonemes(text, pipe)
            text_ids = phonemes_to_st2_ids(phonemes, device)
            asr, F0, Ncurve, mel_even = prepare_asr_f0_n(st2, mel.squeeze(0), text_ids)
            mel_crop = crop_mel_for_style(mel_even, max_frames=192, min_frames=64)

            s_a_tr, _ = enc(mel_crop)
            s_a_off = st2.style_encoder(mel_crop.unsqueeze(1))

            y_tr = decode_with_style(st2, asr, F0, Ncurve, s_a_tr).detach().cpu().numpy().squeeze()
            y_off = decode_with_style(st2, asr, F0, Ncurve, s_a_off).detach().cpu().numpy().squeeze()

            stem = f"{i+1:02d}"
            sf.write(out / f"{stem}_original.wav", wave, SAMPLE_RATE)
            sf.write(out / f"{stem}_st2_official_style.wav", y_off, SAMPLE_RATE)
            sf.write(out / f"{stem}_trained_Ea.wav", y_tr, SAMPLE_RATE)
            html.append(f"<h3>{stem} {text[:80]}</h3>")
            html.append(f"<p>orig</p><audio controls src='{stem}_original.wav'></audio>")
            html.append(f"<p>ST2 official</p><audio controls src='{stem}_st2_official_style.wav'></audio>")
            html.append(f"<p>trained E_a</p><audio controls src='{stem}_trained_Ea.wav'></audio>")
            print(f"{stem} ok s_a_tr={float(s_a_tr.norm()):.3f} s_a_off={float(s_a_off.norm()):.3f}", flush=True)

    html.append("</body></html>")
    (out / "index.html").write_text("\n".join(html), encoding="utf-8")
    print(f"DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
