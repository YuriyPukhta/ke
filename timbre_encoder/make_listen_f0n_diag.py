"""
Diagnostic listen: Exp C (N scale) + Exp B oracle (calibrated F0/N from ref wav).

Uses frozen opt timbre + baseline prosody + forced d_gt.

  .venv\\Scripts\\python.exe -m timbre_encoder.make_listen_f0n_diag
"""
from __future__ import annotations

import argparse
import html as html_lib
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

from kokoro.pipeline import KPipeline  # noqa: E402
from kokoro_revers.model import KModel  # noqa: E402

from timbre_encoder.librispeech_dataset import parse_d_gt  # noqa: E402
from timbre_encoder.mel_preprocess import SAMPLE_RATE  # noqa: E402
from timbre_encoder.optimize_speaker_ref_s import (  # noqa: E402
    build_alignment,
    extract_f0_n,
    kokoro_ids,
    load_speaker_rows,
    load_timbre_ckpt,
    load_wav,
    make_ref_s,
    pack_slot,
)


def match_mean_std(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Map src contour into ref's mean/std (Kokoro unit calibration)."""
    s = src.float().view(-1)
    r = ref.float().view(-1).detach()
    if s.numel() != r.numel():
        s = F.interpolate(s.view(1, 1, -1), size=r.numel(), mode="linear", align_corners=False).view(-1)
    s_std = s.std().clamp_min(1e-5)
    r_std = r.std().clamp_min(1e-5)
    return (s - s.mean()) / s_std * r_std + r.mean()


def rms(x: torch.Tensor) -> float:
    return float(x.float().pow(2).mean().sqrt())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", type=str, default="4195")
    ap.add_argument(
        "--timbre-ckpt",
        type=str,
        default=str(ROOT / "timbre_encoder/runs/speaker_ref_s_4195/timbre.pt"),
    )
    ap.add_argument("--baseline", type=str, default="af_heart")
    ap.add_argument(
        "--manifest",
        type=str,
        default=str(ROOT / "timbre_encoder/manifests/train-clean-100.filtered.csv"),
    )
    ap.add_argument("--holdout", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-phonemes", type=int, default=100)
    ap.add_argument("--max-wav-sec", type=float, default=6.0)
    ap.add_argument("--n-scales", type=str, default="1,1.5,2,3")
    ap.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "timbre_encoder/runs/speaker_ref_s_4195/listen_f0n_diag"),
    )
    args = ap.parse_args()

    scales = [float(x) for x in args.n_scales.split(",") if x.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} out={out}", flush=True)

    rows = load_speaker_rows(
        Path(args.manifest),
        args.speaker,
        max_phonemes=args.max_phonemes,
        max_clips=args.holdout + 8,
        seed=args.seed,
    )
    holdout = rows[: args.holdout]
    if not holdout:
        raise SystemExit("no holdout rows")

    pipeline = KPipeline(lang_code="a", model=False)
    baseline_pack = pipeline.load_voice(args.baseline).to(device)
    if baseline_pack.dim() == 2:
        baseline_pack = baseline_pack.unsqueeze(1)

    model = KModel(repo_id="hexgrad/Kokoro-82M").to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    timbre = load_timbre_ckpt(Path(args.timbre_ckpt), device)

    lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>F0/N diagnostic</title>",
        "<style>body{font-family:sans-serif;max-width:960px;margin:1.5rem auto}"
        "audio{width:100%;margin:.2rem 0}.card{margin:1.2rem 0;padding:1rem;border:1px solid #ccc}"
        "code{background:#f4f4f4;padding:0 .25rem}</style></head><body>",
        "<h1>Exp C (N scale) + Exp B (oracle F0/N)</h1>",
        f"<p>timbre=<code>{html_lib.escape(args.timbre_ckpt)}</code> "
        f"baseline={args.baseline} speaker={args.speaker}</p>",
        "<p><b>opt N×1</b> = whisper candidate; "
        "<b>N×k</b> = Exp C; "
        "<b>oracle</b> = F0/N contour from ref wav, mean/std-matched to "
        "<i>baseline</i> Kokoro pred (not whisper scale).</p>",
    ]

    with torch.no_grad():
        for i, row in enumerate(holdout):
            phonemes = (row.get("phonemes_st2") or "").strip()
            d_gt = parse_d_gt(row.get("d_gt"))
            if not phonemes or not d_gt:
                continue
            input_ids, ids, dropped = kokoro_ids(model, phonemes, device)
            if dropped or len(ids) != len(phonemes):
                print(f"#{i} skip tokens dropped={dropped}", flush=True)
                continue
            aln = build_alignment(d_gt, device)
            if aln.shape[1] != input_ids.shape[1]:
                print(f"#{i} skip align len", flush=True)
                continue

            wave = load_wav(row["audio_path"], args.max_wav_sec)
            slot = pack_slot(baseline_pack, len(phonemes))
            opt_ref = make_ref_s(timbre, slot[0, 128:]).to(device)
            base_ref = slot.to(device)

            text = html_lib.escape((row.get("text") or "")[:160])
            block = [
                f"<div class='card'><h3>#{i} spk={row['speaker_id']}</h3>",
                f"<p>{text}</p>",
            ]

            def save(label: str, audio: torch.Tensor, name: str) -> None:
                path = out / name
                a = audio.detach().cpu().float()
                if a.dim() > 1:
                    a = a.squeeze()
                sf.write(str(path), a.numpy(), SAMPLE_RATE)
                block.append(
                    f"<p>{html_lib.escape(label)} "
                    f"<small>rms={rms(a):.4f}</small></p>"
                    f"<audio controls src='{path.name}'></audio>"
                )

            save("ref", wave, f"{i:02d}_ref.wav")

            audio_base, _, _, _ = model.forward_with_alignment(input_ids, base_ref, aln)
            save("baseline forced", audio_base, f"{i:02d}_baseline.wav")

            audio_1, _, _, _ = model.forward_with_alignment(
                input_ids, opt_ref, aln, n_scale=1.0
            )
            save("opt + N×1 (whisper?)", audio_1, f"{i:02d}_opt_n1.0.wav")

            rms_c = {}
            for k in scales:
                if abs(k - 1.0) < 1e-6:
                    continue
                audio_k, _, _, _ = model.forward_with_alignment(
                    input_ids, opt_ref, aln, n_scale=k
                )
                save(f"opt + N×{k:g} (Exp C)", audio_k, f"{i:02d}_opt_n{k:g}.wav")
                rms_c[k] = rms(audio_k)

            # Calibrate wav contours to *baseline* pred units (not whisper opt)
            _, f0_base, n_base, _ = model.forward_with_alignment(
                input_ids, base_ref, aln, n_scale=1.0
            )
            f0_wav, e_wav = extract_f0_n(wave.numpy())
            f0_b = f0_base.squeeze()
            n_b = n_base.squeeze()
            if f0_b.dim() > 1:
                f0_b = f0_b[0]
            if n_b.dim() > 1:
                n_b = n_b[0]
            f0_inj = match_mean_std(f0_wav.to(device), f0_b)
            n_inj = match_mean_std(e_wav.to(device), n_b)

            audio_or, _, _, _ = model.forward_with_alignment(
                input_ids,
                opt_ref,
                aln,
                f0_override=f0_inj,
                n_override=n_inj,
            )
            save("opt + oracle F0/N (Exp B)", audio_or, f"{i:02d}_opt_oracle.wav")

            audio_bo, _, _, _ = model.forward_with_alignment(
                input_ids,
                base_ref,
                aln,
                f0_override=f0_inj,
                n_override=n_inj,
            )
            save("baseline + oracle F0/N", audio_bo, f"{i:02d}_base_oracle.wav")

            block.append("</div>")
            lines.extend(block)
            print(
                f"#{i} rms ref={rms(wave):.4f} opt_n1={rms(audio_1):.4f} "
                f"c={ {g: round(v,4) for g,v in rms_c.items()} } "
                f"oracle={rms(audio_or):.4f} base={rms(audio_base):.4f}",
                flush=True,
            )

    lines.append("</body></html>")
    (out / "index.html").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
