"""
Path B: build Kokoro-space timbre teachers via timbre-only gradient opt.

Optimizes only ref_s[:128]; prosody frozen from baseline pack.
Forced duration from baseline (same phoneme length slot).

  .venv\\Scripts\\python.exe -m timbre_encoder.build_teacher_dataset --n 12 --epochs 40
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

from kokoro.pipeline import KPipeline
from kokoro_revers.model import KModel

from timbre_encoder.librispeech_dataset import text_to_kokoro_phonemes
from timbre_encoder.voice_pack import assemble_voice_pack

SAMPLE_RATE = 24000


def mel_opt(waveform: torch.Tensor) -> torch.Tensor:
    """Mel used by cloned_voice / demo optimize_ref_s (not StyleTTS2 mel)."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=256,
        n_mels=80,
    ).to(waveform.device)
    mel = mel_transform(waveform)
    return torch.log(torch.clamp(mel, min=1e-5))


def phonemes_to_ids(model: KModel, phonemes: str, device: torch.device) -> torch.Tensor:
    ids = list(filter(lambda i: i is not None, map(lambda p: model.vocab.get(p), phonemes)))
    return torch.LongTensor([[0, *ids, 0]]).to(device)


def pick_baseline(text_hint: str, pipeline: KPipeline) -> str:
    # crude: default female; user can override
    return "af_heart"


@torch.enable_grad()
def optimize_timbre(
    model: KModel,
    input_ids: torch.Tensor,
    target_mel: torch.Tensor,
    baseline_pack: torch.Tensor,
    epochs: int,
    lr: float,
    device: torch.device,
) -> tuple[torch.Tensor, list[float]]:
    """
    Timbre-only Adam on mel L1 with forced duration from baseline slot.
    Returns optimized timbre [128] and loss history.
    """
    n_tok = input_ids.shape[1]
    # duration tokens = phonemes including BOS/EOS already in input_ids
    idx = min(max(n_tok - 1, 0), baseline_pack.shape[0] - 1)
    base_ref = baseline_pack[idx, 0].float().to(device)  # [256]

    with torch.no_grad():
        _, target_dur = model.forward_with_tokens_grad(input_ids, base_ref.unsqueeze(0))

    timbre = base_ref[:128].clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([timbre], lr=lr)
    history: list[float] = []

    for ep in range(epochs):
        opt.zero_grad(set_to_none=True)
        ref_s = torch.cat([timbre, base_ref[128:].detach()], dim=-1).unsqueeze(0)
        audio_pred, _ = model.forward_with_tokens_grad(input_ids, ref_s, target_dur=target_dur)
        if audio_pred.dim() == 1:
            audio_pred = audio_pred.unsqueeze(0)
        pred_mel = mel_opt(audio_pred)
        t = min(pred_mel.shape[-1], target_mel.shape[-1])
        loss = F.l1_loss(pred_mel[..., :t], target_mel[..., :t])
        # stay near Kokoro manifold
        loss = loss + 0.05 * (timbre - base_ref[:128].detach()).pow(2).mean()
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
        if (ep + 1) % 10 == 0 or ep == 0:
            print(
                f"  ep {ep+1:03d}/{epochs} loss={history[-1]:.4f} "
                f"norm={float(timbre.detach().norm()):.3f}",
                flush=True,
            )

    return timbre.detach().cpu(), history


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        default=str(ROOT / "timbre_encoder" / "manifests" / "dev-clean.filtered.csv"),
    )
    ap.add_argument("--out-dir", default=str(ROOT / "timbre_encoder" / "runs" / "teachers_v0"))
    ap.add_argument("--n", type=int, default=12, help="Number of clips")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--max-sec", type=float, default=2.5)
    ap.add_argument("--baseline", default="af_heart")
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = Path(args.out_dir)
    (out / "teachers").mkdir(parents=True, exist_ok=True)
    (out / "listen").mkdir(parents=True, exist_ok=True)

    pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    model = KModel(repo_id="hexgrad/Kokoro-82M").to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    # cudnn RNN backward requires modules in train() even if weights frozen
    import torch.nn as nn

    for m in model.modules():
        if isinstance(m, (nn.LSTM, nn.GRU, nn.RNN)):
            m.train()
    baseline_pack = pipe.load_voice(args.baseline).to(device)
    rows = list(csv.DictReader(open(args.manifest, encoding="utf-8")))

    # Prefer shorter utterances, unique speakers
    scored = []
    for r in rows:
        t = r["text"]
        if 20 <= len(t) <= 80:
            scored.append(r)
    # unique speaker
    seen = set()
    picked = []
    for r in scored:
        sid = r.get("speaker_id", r["audio_path"])
        if sid in seen:
            continue
        seen.add(sid)
        picked.append(r)
        if len(picked) >= args.n:
            break
    if len(picked) < args.n:
        picked = scored[: args.n]

    meta_rows = []
    html = ["<!DOCTYPE html><html><body><h1>Timbre teachers</h1>"]

    for i, row in enumerate(picked):
        text = row["text"]
        wav_path = row["audio_path"]
        print(f"\n=== [{i+1}/{len(picked)}] {Path(wav_path).name} ===", flush=True)
        print(f"text: {text}", flush=True)

        wave, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        wave, _ = librosa.effects.trim(wave, top_db=30)
        wave = wave[: int(args.max_sec * SAMPLE_RATE)]
        if len(wave) < SAMPLE_RATE // 2:
            print("skip too short", flush=True)
            continue

        try:
            phonemes = text_to_kokoro_phonemes(text, pipe)
        except Exception as e:
            print(f"skip g2p: {e}", flush=True)
            continue
        if not (3 <= len(phonemes) <= 200):
            print(f"skip phoneme len {len(phonemes)}", flush=True)
            continue

        input_ids = phonemes_to_ids(model, phonemes, device)
        target = torch.from_numpy(wave.astype(np.float32)).to(device)
        target_mel = mel_opt(target)

        timbre, hist = optimize_timbre(
            model, input_ids, target_mel, baseline_pack, args.epochs, args.lr, device
        )

        teacher_path = out / "teachers" / f"{i+1:02d}_timbre.pt"
        torch.save(
            {
                "timbre": timbre,
                "baseline": args.baseline,
                "audio_path": wav_path,
                "text": text,
                "phonemes": phonemes,
                "speaker_id": row.get("speaker_id", ""),
                "loss_hist": hist,
                "final_loss": hist[-1] if hist else None,
            },
            teacher_path,
        )

        # listen: original vs teacher pack synth
        pack = assemble_voice_pack(timbre, baseline_pack.cpu())
        chunks = []
        for r in pipe(text, voice=pack, speed=1):
            if r.audio is not None:
                chunks.append(r.audio.cpu().numpy())
        synth = np.concatenate(chunks) if chunks else np.zeros(1000)

        stem = f"{i+1:02d}"
        sf.write(out / "listen" / f"{stem}_original.wav", wave, SAMPLE_RATE)
        sf.write(out / "listen" / f"{stem}_teacher_kokoro.wav", synth, SAMPLE_RATE)
        html.append(f"<h3>{stem} loss={hist[-1]:.3f} {text[:60]}</h3>")
        html.append(f"<p>orig</p><audio controls src='{stem}_original.wav'></audio>")
        html.append(f"<p>teacher→Kokoro</p><audio controls src='{stem}_teacher_kokoro.wav'></audio>")

        meta_rows.append(
            {
                "teacher_path": str(teacher_path),
                "audio_path": wav_path,
                "text": text,
                "baseline": args.baseline,
                "final_loss": hist[-1] if hist else None,
                "timbre_norm": float(timbre.norm()),
            }
        )

    (out / "listen" / "index.html").write_text("\n".join(html + ["</body></html>"]), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(meta_rows, indent=2), encoding="utf-8")
    print(f"\nDONE teachers={len(meta_rows)} -> {out}", flush=True)


if __name__ == "__main__":
    main()
