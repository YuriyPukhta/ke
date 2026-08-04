"""
Train ST2 s_a → Kokoro[:128] mapper on official Kokoro voice synth pairs.

Why this can work (narrow claim):
  pairs are Kokoro-in-distribution audio + known pack timbre.
  E_a (frozen, ST2-space) sees that audio; mapper learns to hit Kokoro teacher.
  Generalization to real LibriTTS is the open question — listen-test checks it.

  .venv\\Scripts\\python.exe -m timbre_encoder.train_st2_to_kokoro_mapper --steps 2000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kokoro.pipeline import KPipeline

from timbre_encoder.dual_encoder import crop_mel_for_style
from timbre_encoder.mel_preprocess import waveform_to_mel
from timbre_encoder.st2_kokoro_mapper import St2ToKokoroMapper
from timbre_encoder.st2_runtime import load_frozen_st2
from timbre_encoder.train_encoders_styletts2 import AbsoluteDualStyleEncoder

DEFAULT_VOICES = [
    # US/UK English packs — more diversity for mapper
    "af_heart", "af_bella", "af_sarah", "af_alloy", "af_nova", "af_sky",
    "af_jessica", "af_nicole", "af_kore", "af_aoede", "af_river",
    "am_michael", "am_adam", "am_fenrir", "am_puck", "am_echo", "am_eric",
    "am_liam", "am_onyx", "am_santa",
    "bf_emma", "bf_alice", "bf_isabella", "bf_lily",
    "bm_george", "bm_daniel", "bm_fable", "bm_lewis",
]

TEXTS = [
    "Hello, this is a short voice sample for timbre encoding.",
    "The quick brown fox jumps over the lazy dog near the river.",
    "Please remember to breathe, smile, and speak clearly today.",
    "In the middle of difficulty lies opportunity and calm focus.",
    "She sold seashells by the seashore and sang softly.",
    "Numbers one through ten are easy to say with steady pace.",
]

# Sample teacher timbre from several length slots (packs differ slightly by length).
TEACHER_SLOTS = [40, 80, 120, 200]


def _baseline_name(voice: str) -> str:
    if voice.startswith(("af_", "bf_")):
        return "af_heart"
    return "am_michael"


def synth_voice(pipeline: KPipeline, text: str, voice_name: str) -> torch.Tensor:
    chunks = []
    for result in pipeline(text, voice=voice_name, speed=1):
        if result.audio is not None:
            chunks.append(result.audio.detach().cpu())
    if not chunks:
        raise RuntimeError(f"No audio for {voice_name}")
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def build_cache(
    pipeline: KPipeline,
    ea: AbsoluteDualStyleEncoder,
    voices: list[str],
    device: torch.device,
    style_frames: int,
) -> list[dict]:
    ea.eval()
    cache: list[dict] = []
    packs = {
        "af_heart": pipeline.load_voice("af_heart"),
        "am_michael": pipeline.load_voice("am_michael"),
    }
    for v in voices:
        try:
            pack = pipeline.load_voice(v)
        except Exception as e:
            print(f"[skip voice] {v}: {e}", flush=True)
            continue
        base_name = _baseline_name(v)
        baseline_timbre = packs[base_name][100, 0, :128].float()
        for ti, text in enumerate(TEXTS):
            print(f"cache {v} text{ti}", flush=True)
            wav = synth_voice(pipeline, text, v)
            mel = waveform_to_mel(wav, device=device)
            mel_crop = crop_mel_for_style(mel, max_frames=style_frames, min_frames=64)
            s_a, _ = ea(mel_crop)
            s_a_cpu = s_a.squeeze(0).cpu()
            for slot in TEACHER_SLOTS:
                teacher = pack[slot, 0, :128].float()
                cache.append(
                    {
                        "voice": v,
                        "text": text,
                        "slot": slot,
                        "s_a": s_a_cpu,
                        "teacher": teacher.cpu(),
                        "baseline_timbre": baseline_timbre.cpu(),
                        "baseline_name": base_name,
                    }
                )
    return cache


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}", flush=True)

    print("Loading frozen ST2 + E_a...", flush=True)
    st2, _ = load_frozen_st2(device)
    ea = AbsoluteDualStyleEncoder.from_st2(st2).to(device)
    ckpt = torch.load(args.ea_ckpt, map_location=device, weights_only=False)
    ea.load_state_dict(ckpt["encoders"])
    ea.eval()
    for p in ea.parameters():
        p.requires_grad = False

    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    cache = build_cache(pipeline, ea, voices, device, args.style_frames)
    if not cache:
        raise RuntimeError("Empty cache")
    print(f"cache size={len(cache)}", flush=True)

    # Free ST2 GPU memory — mapper training only needs small tensors
    del st2, ea
    if device.type == "cuda":
        torch.cuda.empty_cache()

    mapper = St2ToKokoroMapper(
        residual_scale=args.residual_scale,
        mode=args.mode,
        hidden=args.hidden,
    ).to(device)
    mapper.train()
    opt = torch.optim.Adam(mapper.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []

    for step in range(1, args.steps + 1):
        item = cache[int(np.random.randint(0, len(cache)))]
        s_a = item["s_a"].unsqueeze(0).to(device)
        teacher = item["teacher"].unsqueeze(0).to(device)
        baseline = item["baseline_timbre"].unsqueeze(0).to(device)

        pred = mapper(s_a, baseline if args.mode == "residual" else None)
        loss_l1 = F.l1_loss(pred, teacher)
        loss_cos = 1.0 - F.cosine_similarity(pred, teacher, dim=-1).mean()
        loss = loss_l1 + args.lambda_cos * loss_cos

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        row = {
            "step": step,
            "loss": float(loss.detach()),
            "l1": float(loss_l1.detach()),
            "cos": float(loss_cos.detach()),
            "voice": item["voice"],
        }
        history.append(row)
        if step % args.log_every == 0 or step == 1:
            print(
                f"step={step:05d} loss={row['loss']:.4f} l1={row['l1']:.4f} "
                f"cos={row['cos']:.4f} | {item['voice']}",
                flush=True,
            )
        if step % args.save_every == 0 or step == args.steps:
            path = out_dir / ("mapper_final.pt" if step == args.steps else f"mapper_step{step:05d}.pt")
            torch.save(
                {
                    "step": step,
                    "mapper": mapper.state_dict(),
                    "residual_scale": args.residual_scale,
                    "mode": args.mode,
                    "hidden": args.hidden,
                    "ea_ckpt": str(args.ea_ckpt),
                    "args": vars(args),
                },
                path,
            )
            print(f"saved {path}", flush=True)

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"DONE -> {out_dir / 'mapper_final.pt'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ea-ckpt",
        default=str(ROOT / "timbre_encoder" / "runs" / "st2_encoders_v2" / "encoders_final.pt"),
    )
    ap.add_argument("--voices", default=",".join(DEFAULT_VOICES))
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-cos", type=float, default=0.2)
    ap.add_argument("--residual-scale", type=float, default=0.9)
    ap.add_argument("--mode", choices=["residual", "absolute"], default="residual")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--style-frames", type=int, default=192)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument(
        "--out-dir",
        default=str(ROOT / "timbre_encoder" / "runs" / "st2_kokoro_mapper_v2"),
    )
    ap.add_argument("--device", default="")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
