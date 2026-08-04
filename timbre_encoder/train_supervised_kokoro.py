"""
Supervised DualStyleEncoder in Kokoro voice space.

Teacher = official voice pack timbre. Audio = Kokoro synthesis of that voice.
This is learnable (unlike matching LibriTTS mel through frozen Kokoro).

  python -m timbre_encoder.train_supervised_kokoro --steps 500
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

from kokoro.pipeline import KPipeline

from timbre_encoder.dual_encoder import DualStyleEncoder, crop_mel_for_style
from timbre_encoder.mel_preprocess import SAMPLE_RATE, waveform_to_mel


DEFAULT_VOICES = [
    "af_heart",
    "af_bella",
    "af_sarah",
    "af_alloy",
    "am_michael",
    "am_adam",
    "am_fenrir",
    "am_puck",
    "bf_emma",
    "bm_george",
]

TEXTS = [
    "Hello, this is a short voice sample for timbre encoding.",
    "The quick brown fox jumps over the lazy dog near the river.",
    "Please remember to breathe, smile, and speak clearly today.",
]


def synth_voice(pipeline: KPipeline, text: str, voice_name: str) -> torch.Tensor:
    chunks = []
    for result in pipeline(text, voice=voice_name, speed=1):
        if result.audio is not None:
            chunks.append(result.audio.detach().cpu())
    if not chunks:
        raise RuntimeError(f"No audio for {voice_name}")
    return torch.cat(chunks, dim=0)


def build_cache(pipeline: KPipeline, voices: list[str], device: torch.device) -> list[dict]:
    cache: list[dict] = []
    heart_pack = pipeline.load_voice("af_heart")
    heart_ref = heart_pack[100, 0].float()  # fixed residual anchor
    for v in voices:
        try:
            pack = pipeline.load_voice(v)
        except Exception as e:
            print(f"[skip voice] {v}: {e}", flush=True)
            continue
        teacher = pack[100, 0, :128].float()
        for ti, text in enumerate(TEXTS):
            print(f"cache {v} text{ti}", flush=True)
            wav = synth_voice(pipeline, text, v)
            mel = waveform_to_mel(wav, device=device).squeeze(0).cpu()
            cache.append(
                {
                    "voice": v,
                    "text": text,
                    "mel": mel,
                    "teacher": teacher,
                    "baseline_ref": heart_ref,  # always residual from af_heart
                    "waveform": wav,
                }
            )
    return cache


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}", flush=True)
    pipeline = KPipeline(lang_code="a")
    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    cache = build_cache(pipeline, voices, device)
    if not cache:
        raise RuntimeError("Empty cache — no voices loaded")
    print(f"cache size={len(cache)}", flush=True)

    enc = DualStyleEncoder().to(device)
    enc.train()
    opt = torch.optim.Adam(enc.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps + 1):
        item = cache[np.random.randint(0, len(cache))]
        mel = crop_mel_for_style(item["mel"].unsqueeze(0).to(device), max_frames=args.style_frames)
        baseline = item["baseline_ref"].unsqueeze(0).to(device)
        teacher = item["teacher"].unsqueeze(0).to(device)

        s_a, _ = enc(mel, baseline, timbre_only=True)
        loss_l1 = F.l1_loss(s_a, teacher)
        loss_cos = 1.0 - F.cosine_similarity(s_a, teacher, dim=-1).mean()
        loss = loss_l1 + 0.1 * loss_cos

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % args.log_every == 0 or step == 1:
            print(
                f"step {step:05d} | loss={loss.item():.4f} l1={loss_l1.item():.4f} "
                f"cos={loss_cos.item():.4f} | {item['voice']}",
                flush=True,
            )
        if step % args.save_every == 0 or step == args.steps:
            ckpt = out_dir / (f"dual_encoder_step{step:05d}.pt" if step < args.steps else "dual_encoder_final.pt")
            torch.save(
                {
                    "step": step,
                    "encoders": enc.state_dict(),
                    "residual_scale": enc.residual_scale,
                    "baseline": "af_heart",
                    "mode": "supervised_kokoro",
                    "config": vars(args),
                },
                ckpt,
            )
            print(f"saved {ckpt}", flush=True)

    print(f"done -> {out_dir / 'dual_encoder_final.pt'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", default=",".join(DEFAULT_VOICES))
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--style-frames", type=int, default=192)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--out-dir", default=str(ROOT / "timbre_encoder" / "runs" / "supervised_kokoro"))
    ap.add_argument("--device", default="")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
