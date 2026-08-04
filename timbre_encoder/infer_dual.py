"""
Infer: reference wav -> residual DualStyleEncoder -> Kokoro synth.

  python -m timbre_encoder.infer_dual --wav path.wav --text "Hello" --ckpt dual_encoder_final.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

from kokoro.pipeline import KPipeline

from timbre_encoder.dual_encoder import DualStyleEncoder
from timbre_encoder.voice_pack import assemble_voice_pack


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="Reference audio")
    ap.add_argument("--text", required=True)
    ap.add_argument("--ckpt", required=True, help="dual_encoder_*.pt")
    ap.add_argument("--baseline", default="", help="Override baseline voice (default: from ckpt or af_heart)")
    ap.add_argument("--out", default="timbre_encoder/runs/infer_out.wav")
    ap.add_argument("--device", default="")
    ap.add_argument(
        "--mode",
        choices=["full", "timbre_only"],
        default="timbre_only",
    )
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    baseline_name = args.baseline or ckpt.get("baseline") or "af_heart"
    scale = float(ckpt.get("residual_scale", 0.25))

    encoders = DualStyleEncoder().to(device)
    encoders.residual_scale = scale
    encoders.load_state_dict(ckpt["encoders"])
    encoders.eval()

    pipeline = KPipeline(lang_code="a")
    baseline_pack = pipeline.load_voice(baseline_name).to(device)
    baseline_ref = baseline_pack[100, 0].unsqueeze(0)

    with torch.no_grad():
        ref_s_full = encoders.encode_wav(
            args.wav,
            baseline_ref_s=baseline_ref,
            device=device,
            timbre_only=(args.mode == "timbre_only"),
        )
        s_a = ref_s_full[:, :128]
        pack = assemble_voice_pack(s_a, baseline_pack)
        if args.mode == "full":
            s_p = ref_s_full[:, 128:]
            pack[:, :, 128:] = 0.85 * pack[:, :, 128:] + 0.15 * s_p.view(1, 1, 128)

        audio_chunks = []
        for result in pipeline(args.text, voice=pack, speed=1):
            if result.audio is not None:
                audio_chunks.append(result.audio.cpu().numpy())
        if not audio_chunks:
            raise RuntimeError("No audio generated")
        audio = np.concatenate(audio_chunks, axis=0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), audio, 24000)
    print(
        f"wrote {out} ({len(audio)/24000:.2f}s) mode={args.mode} "
        f"baseline={baseline_name} scale={scale} "
        f"sa_norm={s_a.norm().item():.3f} base_norm={baseline_ref[0,:128].norm().item():.3f}"
    )


if __name__ == "__main__":
    main()
