"""
Fine-tune ST2→Kokoro mapper on path-B teachers (LibriTTS → timbre-only opt).

Pairs: s_a = frozen E_a(mel_ref), y = teacher timbre from build_teacher_dataset.

  .venv\\Scripts\\python.exe -m timbre_encoder.train_mapper_on_teachers
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kokoro.pipeline import KPipeline

from timbre_encoder.dual_encoder import crop_mel_for_style
from timbre_encoder.mel_preprocess import SAMPLE_RATE, waveform_to_mel
from timbre_encoder.st2_kokoro_mapper import St2ToKokoroMapper
from timbre_encoder.st2_runtime import load_frozen_st2
from timbre_encoder.train_encoders_styletts2 import AbsoluteDualStyleEncoder
from timbre_encoder.voice_pack import assemble_voice_pack


def build_pairs(
    teachers_manifest: Path,
    ea: AbsoluteDualStyleEncoder,
    device: torch.device,
    style_frames: int,
) -> list[dict]:
    meta = json.loads(teachers_manifest.read_text(encoding="utf-8"))
    pairs = []
    ea.eval()
    with torch.no_grad():
        for row in meta:
            # demo-mel L1 starts ~5; keep if improved vs first epoch and not exploding
            if row.get("final_loss") is not None and row["final_loss"] > 4.5:
                print(f"skip high loss {row['final_loss']:.2f}", flush=True)
                continue
            tdata = torch.load(row["teacher_path"], map_location="cpu", weights_only=False)
            wave, _ = librosa.load(row["audio_path"], sr=SAMPLE_RATE, mono=True)
            wave, _ = librosa.effects.trim(wave, top_db=30)
            wave = wave[: int(2.5 * SAMPLE_RATE)]
            mel = waveform_to_mel(torch.from_numpy(wave.astype(np.float32)), device=device)
            mel_crop = crop_mel_for_style(mel, max_frames=style_frames, min_frames=64)
            s_a, _ = ea(mel_crop)
            pairs.append(
                {
                    "s_a": s_a.squeeze(0).cpu(),
                    "teacher": tdata["timbre"].float().cpu(),
                    "baseline": row["baseline"],
                    "audio_path": row["audio_path"],
                    "text": row["text"],
                    "teacher_path": row["teacher_path"],
                }
            )
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--teachers",
        default=str(ROOT / "timbre_encoder" / "runs" / "teachers_v0" / "manifest.json"),
    )
    ap.add_argument(
        "--ea-ckpt",
        default=str(ROOT / "timbre_encoder" / "runs" / "st2_encoders_v2" / "encoders_final.pt"),
    )
    ap.add_argument(
        "--init-mapper",
        default=str(ROOT / "timbre_encoder" / "runs" / "st2_kokoro_mapper_v2" / "mapper_final.pt"),
        help="Warm-start from Kokoro-voice mapper",
    )
    ap.add_argument(
        "--out-dir",
        default=str(ROOT / "timbre_encoder" / "runs" / "mapper_teachers_v0"),
    )
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--residual-scale", type=float, default=0.9)
    ap.add_argument("--style-frames", type=int, default=192)
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading E_a...", flush=True)
    st2, _ = load_frozen_st2(device)
    ea = AbsoluteDualStyleEncoder.from_st2(st2).to(device)
    ea.load_state_dict(torch.load(args.ea_ckpt, map_location=device, weights_only=False)["encoders"])
    ea.eval()
    for p in ea.parameters():
        p.requires_grad = False

    pairs = build_pairs(Path(args.teachers), ea, device, args.style_frames)
    print(f"pairs={len(pairs)}", flush=True)
    if len(pairs) < 2:
        raise RuntimeError("Need more teachers")

    del st2
    if device.type == "cuda":
        torch.cuda.empty_cache()

    init = torch.load(args.init_mapper, map_location=device, weights_only=False)
    mapper = St2ToKokoroMapper(
        residual_scale=args.residual_scale,
        mode=init.get("mode", "residual"),
        hidden=init.get("hidden", 512),
    ).to(device)
    mapper.load_state_dict(init["mapper"])
    mapper.train()
    opt = torch.optim.Adam(mapper.parameters(), lr=args.lr)

    pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    baselines = {
        "af_heart": pipe.load_voice("af_heart")[100, 0, :128].float(),
        "am_michael": pipe.load_voice("am_michael")[100, 0, :128].float(),
    }

    for step in range(1, args.steps + 1):
        item = pairs[int(np.random.randint(0, len(pairs)))]
        s_a = item["s_a"].unsqueeze(0).to(device)
        teacher = item["teacher"].unsqueeze(0).to(device)
        base = baselines.get(item["baseline"], baselines["af_heart"]).unsqueeze(0).to(device)
        # reshape teacher to [1,128]
        if teacher.numel() != 128:
            teacher = teacher.reshape(1, -1)[:, :128]

        pred = mapper(s_a, base)
        loss_l1 = F.l1_loss(pred, teacher)
        loss_cos = 1.0 - F.cosine_similarity(pred, teacher, dim=-1).mean()
        loss = loss_l1 + 0.2 * loss_cos

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 50 == 0 or step == 1:
            print(
                f"step={step:04d} loss={float(loss):.4f} l1={float(loss_l1):.4f} cos={float(loss_cos):.4f}",
                flush=True,
            )

    ckpt = {
        "step": args.steps,
        "mapper": mapper.state_dict(),
        "residual_scale": args.residual_scale,
        "mode": init.get("mode", "residual"),
        "hidden": init.get("hidden", 512),
        "ea_ckpt": args.ea_ckpt,
        "teachers": args.teachers,
    }
    torch.save(ckpt, out / "mapper_final.pt")
    print(f"saved {out / 'mapper_final.pt'}", flush=True)

    # quick listen on teacher clips
    listen = out / "listen_test"
    listen.mkdir(exist_ok=True)
    mapper.eval()
    html = ["<!DOCTYPE html><html><body><h1>Mapper on teachers</h1>"]
    base_pack = pipe.load_voice("af_heart")
    base_timbre = baselines["af_heart"].to(device)

    # reload ea for listen (need ST2 again)
    st2, _ = load_frozen_st2(device)
    ea = AbsoluteDualStyleEncoder.from_st2(st2).to(device)
    ea.load_state_dict(torch.load(args.ea_ckpt, map_location=device, weights_only=False)["encoders"])
    ea.eval()

    with torch.no_grad():
        for i, item in enumerate(pairs[: min(5, len(pairs))]):
            wave, _ = librosa.load(item["audio_path"], sr=SAMPLE_RATE, mono=True)
            wave, _ = librosa.effects.trim(wave, top_db=30)
            wave = wave[: int(2.5 * SAMPLE_RATE)]
            mel = waveform_to_mel(torch.from_numpy(wave.astype(np.float32)), device=device)
            mel_crop = crop_mel_for_style(mel, max_frames=192, min_frames=64)
            s_a, _ = ea(mel_crop)
            timbre = mapper(s_a, base_timbre).squeeze(0).cpu()
            pack = assemble_voice_pack(timbre, base_pack)
            chunks = []
            for r in pipe(item["text"], voice=pack, speed=1):
                if r.audio is not None:
                    chunks.append(r.audio.cpu().numpy())
            synth = np.concatenate(chunks)

            # teacher pack for comparison
            tpack = assemble_voice_pack(item["teacher"], base_pack)
            tchunks = []
            for r in pipe(item["text"], voice=tpack, speed=1):
                if r.audio is not None:
                    tchunks.append(r.audio.cpu().numpy())
            tsynth = np.concatenate(tchunks)

            stem = f"{i+1:02d}"
            sf.write(listen / f"{stem}_original.wav", wave, SAMPLE_RATE)
            sf.write(listen / f"{stem}_teacher_opt.wav", tsynth, SAMPLE_RATE)
            sf.write(listen / f"{stem}_mapped.wav", synth, SAMPLE_RATE)
            html.append(f"<h3>{stem} {item['text'][:60]}</h3>")
            html.append(f"<p>orig</p><audio controls src='{stem}_original.wav'></audio>")
            html.append(f"<p>teacher opt</p><audio controls src='{stem}_teacher_opt.wav'></audio>")
            html.append(f"<p>E_a→mapper</p><audio controls src='{stem}_mapped.wav'></audio>")
            print(f"{stem} l1_map_vs_teacher={float((timbre - item['teacher']).abs().mean()):.4f}", flush=True)

    (listen / "index.html").write_text("\n".join(html + ["</body></html>"]), encoding="utf-8")
    print(f"DONE listen -> {listen}", flush=True)


if __name__ == "__main__":
    main()
