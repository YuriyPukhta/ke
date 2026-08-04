"""Diagnose whisper on E_a + baseline: native F0/N vs ST2-cal override."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

from kokoro.pipeline import KPipeline
from kokoro_revers.model import KModel

from timbre_encoder.dual_encoder import DualStyleEncoder, crop_mel_for_style
from gt_cache import feature_path, load_curve
from gt_cache.utils import DEFAULT_FEATURES_ROOT
from timbre_encoder.kokoro_embed_train.common import (
    calibrate_st2_to_kokoro,
    freeze_module,
    kokoro_f0n_for_ref,
    load_ckpt,
    pack_slot,
    prepare_utt,
    to_1d,
)
from timbre_encoder.librispeech_dataset import parse_d_gt
from timbre_encoder.mel_preprocess import SAMPLE_RATE, waveform_to_mel

OUT = Path(__file__).resolve().parent / "results" / "listen" / "stage1_whisper_diag"


def load_wav(path: str) -> np.ndarray:
    w, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    w, _ = librosa.effects.trim(w, top_db=30)
    return w[: int(8 * SAMPLE_RATE)].astype(np.float32)


def rms(x) -> float:
    x = np.asarray(x).reshape(-1)
    return float(np.sqrt(np.mean(x**2) + 1e-12))


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)

    pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    model = KModel(repo_id="hexgrad/Kokoro-82M").to(device)
    freeze_module(model)
    pack = pipe.load_voice("af_heart").to(device)

    ckpt = load_ckpt(
        ROOT / "timbre_encoder/runs/kokoro_embed/stage1/stage1_final.pt",
        map_location=device,
    )
    enc = DualStyleEncoder().to(device)
    enc.load_state_dict(ckpt["encoders"])
    enc.residual_scale = float(ckpt.get("residual_scale", 0.25))
    enc.eval()

    rows: list[dict] = []
    seen: set[str] = set()
    with open(
        ROOT / "timbre_encoder/manifests/dev-clean.filtered.csv", encoding="utf-8"
    ) as f:
        for r in csv.DictReader(f):
            if r.get("align_status") != "ok":
                continue
            if r["speaker_id"] in seen:
                continue
            seen.add(r["speaker_id"])
            rows.append(r)
            if len(rows) >= 4:
                break

    lines = ["stem | cond | wav_rms | N_mean | N_std | F0_voiced_mean"]
    cards: list[str] = []

    for i, r in enumerate(rows):
        ph = (r.get("phonemes_st2") or "").strip()
        d = parse_d_gt(r.get("d_gt"))
        prep = prepare_utt(model, ph, d, device)
        if prep is None:
            continue
        ids, aln, nph = prep
        wav = load_wav(r["audio_path"])
        mel = waveform_to_mel(torch.from_numpy(wav), device=device).squeeze(0)
        base = pack_slot(pack, nph)
        crop = crop_mel_for_style(mel.unsqueeze(0), max_frames=192, min_frames=64)
        with torch.no_grad():
            sa, sp = enc(crop, base, timbre_only=True)
        ref_ea = torch.cat([sa, sp], dim=-1)
        dsa = (sa - base[:, :128]).detach().cpu().view(-1)
        print(
            f"spk={r['speaker_id']} delta_mean={float(dsa.mean()):.4f} "
            f"delta_std={float(dsa.std()):.4f} delta_max={float(dsa.abs().max()):.4f} "
            f"sa_norm={float(sa.norm()):.3f} base_t={float(base[:, :128].norm()):.3f}",
            flush=True,
        )

        f0s = load_curve(feature_path(r["audio_path"], "f0", DEFAULT_FEATURES_ROOT))
        ns = load_curve(feature_path(r["audio_path"], "n", DEFAULT_FEATURES_ROOT))
        f0r, nr = kokoro_f0n_for_ref(model, ids, base, aln)
        f0c, nc = calibrate_st2_to_kokoro(f0s.to(device), ns.to(device), f0r, nr)
        print(
            f"  N: kokoro mean={float(nr.mean()):.3f} cal mean={float(nc.mean()):.3f} "
            f"st2 mean={float(to_1d(ns).mean()):.3f}",
            flush=True,
        )

        stem = f"{i:02d}_{r['speaker_id']}"
        sf.write(str(OUT / f"{stem}_0_original.wav"), wav, SAMPLE_RATE)
        card = [
            f"<div class='card'><h3>{stem}</h3>",
            f"<p>{(r.get('text') or '')[:140]}</p>",
            f"<p>original rms={rms(wav):.4f}</p>",
            f"<audio controls src='{stem}_0_original.wav'></audio>",
        ]

        conds = [
            ("1_baseline_native", base, None, None),
            ("2_baseline_f0n_cal", base, f0c, nc),
            ("3_Ea_base_native", ref_ea, None, None),
            ("4_Ea_base_f0n_cal", ref_ea, f0c, nc),  # make_listen default
            ("5_Ea_base_f0n_kok", ref_ea, f0r, nr),
        ]
        for name, ref, fo, no in conds:
            with torch.no_grad():
                audio, f0u, nu, _ = model.forward_with_alignment(
                    ids, ref, aln, f0_override=fo, n_override=no
                )
            a = audio.detach().cpu().numpy().reshape(-1)
            sf.write(str(OUT / f"{stem}_{name}.wav"), a, SAMPLE_RATE)
            nu1 = to_1d(nu)
            fv = to_1d(f0u)
            fvm = float(fv[fv > 50].mean()) if (fv > 50).any() else 0.0
            line = (
                f"{stem} | {name} | rms={rms(a):.4f} | "
                f"Nmean={float(nu1.mean()):.3f} Nstd={float(nu1.std()):.3f} | F0v={fvm:.1f}"
            )
            print(line, flush=True)
            lines.append(line)
            card.append(
                f"<p><b>{name}</b> rms={rms(a):.4f} Nmean={float(nu1.mean()):.2f}</p>"
                f"<audio controls src='{stem}_{name}.wav'></audio>"
            )
        card.append("</div>")
        cards.append("\n".join(card))

    (OUT / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "index.html").write_text(
        "<!DOCTYPE html><html><head><meta charset='utf-8'><style>"
        "body{font-family:sans-serif;max-width:900px;margin:1.5rem auto}"
        ".card{margin:1rem 0;padding:1rem;border:1px solid #ccc}audio{width:100%}"
        "</style></head><body>"
        "<h1>Whisper diag (E_a + baseline)</h1>"
        "<p>1 baseline native · 2 baseline+ST2 cal F0/N · 3 E_a native F0/N · "
        "4 E_a+cal (make_listen) · 5 E_a+kokoro F0/N</p>"
        + "\n".join(cards)
        + "</body></html>",
        encoding="utf-8",
    )
    print(f"-> {OUT / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
