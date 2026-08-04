"""
Train absolute TimbreEncoder on LibriTTS → good speaker_ref teachers.

No DualStyle residual (0.25·tanh). Mel multi-crop → mean → L1/cos to teacher.
Optional light mel loss through frozen Kokoro (forced d_gt).

  .venv\\Scripts\\python.exe -m timbre_encoder.train_encoder_teachers --steps 1000
"""
from __future__ import annotations

import argparse
import csv
import html as html_lib
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

from kokoro.pipeline import KPipeline  # noqa: E402
from kokoro_revers.model import KModel  # noqa: E402

from timbre_encoder.dual_encoder import MIN_STYLE_FRAMES, crop_mel_for_style  # noqa: E402
from timbre_encoder.encoder import TimbreEncoder  # noqa: E402
from timbre_encoder.librispeech_dataset import parse_d_gt  # noqa: E402
from timbre_encoder.mel_preprocess import SAMPLE_RATE, waveform_to_mel  # noqa: E402
from timbre_encoder.optimize_speaker_ref_s import (  # noqa: E402
    build_alignment,
    kokoro_ids,
    load_wav,
    make_ref_s,
    pack_slot,
)
from timbre_encoder.voice_bank import load_bank  # noqa: E402

# Proven good teachers only (absolute free-vector runs).
DEFAULT_TEACHERS = {
    "6209": ROOT / "timbre_encoder/runs/speaker_ref_s_6209_bankmean_norm/timbre.pt",
    "4195": ROOT / "timbre_encoder/runs/speaker_ref_s_4195/timbre.pt",
}


def load_teacher(path: Path) -> tuple[torch.Tensor, str]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "timbre" in ckpt:
        t = ckpt["timbre"].float().view(-1)
        baseline = str(ckpt.get("baseline") or "af_heart")
    else:
        t = ckpt.float().view(-1)
        baseline = "af_heart"
    if t.numel() != 128:
        raise ValueError(f"bad timbre shape in {path}: {tuple(t.shape)}")
    return t, baseline


def load_speaker_rows(
    manifest: Path,
    speaker: str,
    max_clips: int,
    seed: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with manifest.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("speaker_id") != speaker:
                continue
            if row.get("align_status") != "ok":
                continue
            ph = (row.get("phonemes_st2") or "").strip()
            d_gt = parse_d_gt(row.get("d_gt"))
            if not ph or not d_gt or len(d_gt) != len(ph):
                continue
            if not (2 <= len(ph) <= 100):
                continue
            rows.append(row)
    rng = random.Random(seed)
    rng.shuffle(rows)
    if max_clips > 0:
        rows = rows[:max_clips]
    return rows


def encode_multicrop(
    enc: TimbreEncoder,
    mel: torch.Tensor,
    n_crops: int,
    style_frames: int,
    min_frames: int,
) -> torch.Tensor:
    """mel [80,T] or [1,80,T] → absolute timbre [1,128] via mean of crops."""
    if mel.dim() == 2:
        mel = mel.unsqueeze(0)
    vecs = []
    for _ in range(max(1, n_crops)):
        crop = crop_mel_for_style(mel, max_frames=style_frames, min_frames=min_frames)
        vecs.append(enc.encode_mel(crop))
    return torch.stack(vecs, dim=0).mean(dim=0)


def write_listen(
    out_dir: Path,
    enc: TimbreEncoder,
    model: KModel,
    baseline_pack: torch.Tensor,
    teachers: dict[str, torch.Tensor],
    holdouts: dict[str, list[dict[str, str]]],
    device: torch.device,
    n_crops: int,
    style_frames: int,
    max_wav_sec: float,
) -> None:
    listen = out_dir / "listen_test"
    listen.mkdir(parents=True, exist_ok=True)
    enc.eval()
    lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Encoder teachers</title>",
        "<style>body{font-family:sans-serif;max-width:900px;margin:1.5rem auto}"
        "audio{width:100%}.card{margin:1rem 0;padding:1rem;border:1px solid #ccc}</style></head><body>",
        "<h1>Absolute TimbreEncoder (multi-crop mean)</h1>",
        "<p>ref | teacher forced | encoder forced | baseline pack</p>",
    ]
    with torch.no_grad():
        for spk, rows in holdouts.items():
            teacher = teachers[spk].to(device)
            for i, row in enumerate(rows[:3]):
                ph = (row.get("phonemes_st2") or "").strip()
                d_gt = parse_d_gt(row.get("d_gt"))
                if not ph or not d_gt:
                    continue
                input_ids, ids, dropped = kokoro_ids(model, ph, device)
                if dropped or len(ids) != len(ph):
                    continue
                aln = build_alignment(d_gt, device)
                if aln.shape[1] != input_ids.shape[1]:
                    continue
                wav = load_wav(row["audio_path"], max_wav_sec)
                mel = waveform_to_mel(wav.to(device)).squeeze(0)
                pred = encode_multicrop(enc, mel, n_crops, style_frames, MIN_STYLE_FRAMES)
                slot = pack_slot(baseline_pack, len(ph)).to(device)
                base_p = slot[0, 128:]
                ref_teacher = make_ref_s(teacher, base_p)
                ref_enc = make_ref_s(pred.squeeze(0), base_p)
                ref_base = slot

                def synth(ref_s: torch.Tensor) -> np.ndarray | None:
                    audio, _, _, _ = model.forward_with_alignment(input_ids, ref_s, aln)
                    if audio is None or audio.numel() < 100:
                        return None
                    return audio.detach().cpu().numpy().reshape(-1)

                stem = f"{spk}_{i:02d}"
                sf.write(str(listen / f"{stem}_0_ref.wav"), wav.numpy(), SAMPLE_RATE)
                a_t = synth(ref_teacher)
                a_e = synth(ref_enc)
                a_b = synth(ref_base)
                l1 = float(F.l1_loss(pred.squeeze(0), teacher).cpu())
                cos = float(F.cosine_similarity(pred.squeeze(0), teacher, dim=0).cpu())
                block = [
                    f"<div class='card'><h3>{html_lib.escape(stem)} L1={l1:.4f} cos={cos:.4f}</h3>",
                    f"<p>{html_lib.escape((row.get('text') or '')[:160])}</p>",
                    f"<p>ref</p><audio controls src='{stem}_0_ref.wav'></audio>",
                ]
                if a_t is not None:
                    sf.write(str(listen / f"{stem}_1_teacher.wav"), a_t, SAMPLE_RATE)
                    block.append(
                        f"<p>teacher</p><audio controls src='{stem}_1_teacher.wav'></audio>"
                    )
                if a_e is not None:
                    sf.write(str(listen / f"{stem}_2_encoder.wav"), a_e, SAMPLE_RATE)
                    block.append(
                        f"<p>encoder</p><audio controls src='{stem}_2_encoder.wav'></audio>"
                    )
                if a_b is not None:
                    sf.write(str(listen / f"{stem}_3_baseline.wav"), a_b, SAMPLE_RATE)
                    block.append(
                        f"<p>baseline</p><audio controls src='{stem}_3_baseline.wav'></audio>"
                    )
                block.append("</div>")
                lines.extend(block)
                print(f"listen {stem} L1={l1:.4f} cos={cos:.4f}", flush=True)
    lines.append("</body></html>")
    (listen / "index.html").write_text("\n".join(lines), encoding="utf-8")
    print(f"listen -> {listen / 'index.html'}", flush=True)


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    teachers: dict[str, torch.Tensor] = {}
    teacher_baseline: dict[str, str] = {}
    for spk, path in DEFAULT_TEACHERS.items():
        if args.speakers and spk not in args.speakers.split(","):
            continue
        p = Path(path)
        if not p.is_file():
            print(f"skip missing teacher {p}", flush=True)
            continue
        t, b = load_teacher(p)
        teachers[spk] = t
        teacher_baseline[spk] = b
        print(f"teacher {spk} baseline={b} norm={float(t.norm()):.3f}", flush=True)
    if not teachers:
        raise SystemExit("No teachers loaded")

    # One baseline pack for decode (mean bank if available, else af_heart)
    pipeline = KPipeline(lang_code="a", model=False)
    bank = None
    if Path(args.bank_cache).is_file():
        bank = load_bank(Path(args.bank_cache), device=device)
        baseline_pack = bank.mean_pack.to(device)
        print(f"baseline_pack = voice-bank MEAN ({args.bank_cache})", flush=True)
    else:
        baseline_pack = pipeline.load_voice("af_heart").to(device)
        if baseline_pack.dim() == 2:
            baseline_pack = baseline_pack.unsqueeze(1)
        print("baseline_pack = af_heart", flush=True)

    train_rows: list[tuple[str, dict[str, str]]] = []
    holdouts: dict[str, list[dict[str, str]]] = {}
    for spk in teachers:
        rows = load_speaker_rows(
            Path(args.manifest), spk, max_clips=args.max_clips + args.holdout, seed=args.seed
        )
        if len(rows) < args.holdout + 1:
            raise SystemExit(f"speaker {spk}: need >{args.holdout} clips, got {len(rows)}")
        holdouts[spk] = rows[: args.holdout]
        for r in rows[args.holdout : args.holdout + args.max_clips]:
            train_rows.append((spk, r))
    rng.shuffle(train_rows)
    print(f"train_utt={len(train_rows)} speakers={list(teachers)}", flush=True)

    enc = TimbreEncoder().to(device)
    enc.train()
    opt = torch.optim.Adam(enc.parameters(), lr=args.lr)

    kmodel = None
    if args.lambda_mel > 0:
        kmodel = KModel(repo_id="hexgrad/Kokoro-82M").to(device)
        kmodel.eval()
        for p in kmodel.parameters():
            p.requires_grad_(False)

    history: list[dict] = []
    for step in range(1, args.steps + 1):
        spk, row = train_rows[rng.randrange(len(train_rows))]
        teacher = teachers[spk].to(device)
        try:
            wav = load_wav(row["audio_path"], args.max_wav_sec).to(device)
        except Exception:
            continue
        if wav.numel() < SAMPLE_RATE // 4:
            continue
        mel = waveform_to_mel(wav).squeeze(0)
        if mel.shape[-1] < 16:
            continue

        pred = encode_multicrop(
            enc, mel, args.n_crops, args.style_frames, args.min_style_frames
        )
        loss_l1 = F.l1_loss(pred, teacher.unsqueeze(0))
        loss_cos = 1.0 - F.cosine_similarity(pred, teacher.unsqueeze(0), dim=-1).mean()
        loss = args.lambda_l1 * loss_l1 + args.lambda_cos * loss_cos

        loss_mel = pred.new_zeros(())
        if kmodel is not None and args.lambda_mel > 0:
            ph = (row.get("phonemes_st2") or "").strip()
            d_gt = parse_d_gt(row.get("d_gt"))
            if ph and d_gt:
                input_ids, ids, dropped = kokoro_ids(kmodel, ph, device)
                if not dropped and len(ids) == len(ph):
                    aln = build_alignment(d_gt, device)
                    if aln.shape[1] == input_ids.shape[1]:
                        slot = pack_slot(baseline_pack, len(ph))
                        ref_s = make_ref_s(pred.squeeze(0), slot[0, 128:].detach())
                        audio, _, _, _ = kmodel.forward_with_alignment(input_ids, ref_s, aln)
                        if audio.numel() > 100:
                            pred_mel = waveform_to_mel(
                                audio.view(1, -1), device=device
                            )
                            tgt_mel = waveform_to_mel(wav.unsqueeze(0), device=device)
                            t = min(pred_mel.shape[-1], tgt_mel.shape[-1])
                            if t >= 4:
                                loss_mel = F.l1_loss(
                                    pred_mel[..., :t], tgt_mel[..., :t].detach()
                                )
                                loss = loss + args.lambda_mel * loss_mel

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), args.grad_clip)
        opt.step()

        row_h = {
            "step": step,
            "loss": float(loss.detach()),
            "l1": float(loss_l1.detach()),
            "cos": float(loss_cos.detach()),
            "mel": float(loss_mel.detach()),
            "spk": spk,
        }
        history.append(row_h)
        if step == 1 or step % args.log_every == 0:
            print(
                f"step={step} loss={row_h['loss']:.4f} l1={row_h['l1']:.4f} "
                f"cos={row_h['cos']:.4f} mel={row_h['mel']:.4f} spk={spk}",
                flush=True,
            )
        if step % args.save_every == 0 or step == args.steps:
            ckpt = {
                "step": step,
                "encoder": enc.state_dict(),
                "teachers": {k: str(DEFAULT_TEACHERS[k]) for k in teachers},
                "mode": "absolute_multicrop",
                "args": vars(args),
            }
            path = out_dir / (
                f"encoder_step{step}.pt" if step < args.steps else "encoder_final.pt"
            )
            torch.save(ckpt, path)
            print(f"saved {path}", flush=True)

    # Listen
    if kmodel is None:
        kmodel = KModel(repo_id="hexgrad/Kokoro-82M").to(device)
        kmodel.eval()
        for p in kmodel.parameters():
            p.requires_grad_(False)
    write_listen(
        out_dir,
        enc,
        kmodel,
        baseline_pack,
        teachers,
        holdouts,
        device,
        args.n_crops,
        args.style_frames,
        args.max_wav_sec,
    )
    print(f"DONE -> {out_dir / 'encoder_final.pt'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train absolute TimbreEncoder on teachers")
    ap.add_argument(
        "--manifest",
        default=str(ROOT / "timbre_encoder/manifests/train-clean-100.filtered.csv"),
    )
    ap.add_argument("--speakers", default="6209,4195", help="Comma speaker ids")
    ap.add_argument(
        "--bank-cache",
        default=str(ROOT / "timbre_encoder/runs/voice_bank_us.pt"),
    )
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-clips", type=int, default=64)
    ap.add_argument("--holdout", type=int, default=3)
    ap.add_argument("--max-wav-sec", type=float, default=6.0)
    ap.add_argument("--style-frames", type=int, default=192)
    ap.add_argument("--min-style-frames", type=int, default=MIN_STYLE_FRAMES)
    ap.add_argument("--n-crops", type=int, default=4, help="Multi-crop mean count")
    ap.add_argument("--lambda-l1", type=float, default=1.0)
    ap.add_argument("--lambda-cos", type=float, default=0.1)
    ap.add_argument(
        "--lambda-mel",
        type=float,
        default=0.0,
        help="Optional Kokoro mel loss (0=supervised only)",
    )
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out-dir",
        default=str(ROOT / "timbre_encoder/runs/encoder_teachers_abs"),
    )
    ap.add_argument("--device", default="")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
