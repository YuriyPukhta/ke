"""
Timbre encoder by MEL=MEL reconstruction through frozen Kokoro.

Grad-accum batch; multi-crop mean; absolute TimbreEncoder.
All-GPU on 4GB: --max-phonemes 50 (default). Optional --kokoro-device cpu.

  .venv\\Scripts\\python.exe -m timbre_encoder.train_encoder_recon \\
      --epochs 8 --batch-size 8 --n-crops 4 --max-phonemes 50 \\
      --out-dir timbre_encoder/runs/encoder_recon_all_ok
  # omit --speaker → all align_status=ok, alphabetical
"""
from __future__ import annotations

import argparse
import csv
import html as html_lib
import random
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

from timbre_encoder.dual_encoder import MIN_STYLE_FRAMES, crop_mel_for_style  # noqa: E402
from timbre_encoder.encoder import TimbreEncoder  # noqa: E402
from timbre_encoder.kokoro_embed_train.common import (  # noqa: E402
    freeze_module,
    kokoro_f0n_for_ref,
)
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


def load_rows(
    manifest: Path,
    speaker: str | None,
    max_clips: int,
    max_phonemes: int = 100,
) -> list[dict[str, str]]:
    """align_status=ok rows; sorted by speaker_id, audio_path (alphabetical)."""
    rows: list[dict[str, str]] = []
    with manifest.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if speaker and row.get("speaker_id") != speaker:
                continue
            if row.get("align_status") != "ok":
                continue
            ph = (row.get("phonemes_st2") or "").strip()
            d_gt = parse_d_gt(row.get("d_gt"))
            if not ph or not d_gt or len(d_gt) != len(ph):
                continue
            if not (2 <= len(ph) <= max_phonemes):
                continue
            rows.append(row)
    rows.sort(key=lambda r: (r["speaker_id"], r.get("audio_path") or ""))
    if max_clips > 0:
        rows = rows[:max_clips]
    return rows


def split_holdout_alpha(
    rows: list[dict[str, str]], holdout_per_spk: int, max_holdout: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    holdout: list[dict[str, str]] = []
    train: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    for row in rows:
        spk = row["speaker_id"]
        n = counts.get(spk, 0)
        if n < holdout_per_spk and len(holdout) < max_holdout:
            holdout.append(row)
            counts[spk] = n + 1
        else:
            train.append(row)
    return holdout, train


def encode_multicrop(
    enc: TimbreEncoder,
    mel: torch.Tensor,
    n_crops: int,
    style_frames: int,
    min_frames: int,
) -> torch.Tensor:
    if mel.dim() == 2:
        mel = mel.unsqueeze(0)
    vecs = [
        enc.encode_mel(
            crop_mel_for_style(mel, max_frames=style_frames, min_frames=min_frames)
        )
        for _ in range(max(1, n_crops))
    ]
    return torch.stack(vecs, dim=0).mean(dim=0)


def utt_loss(
    enc: TimbreEncoder,
    kmodel: KModel,
    baseline_pack: torch.Tensor,
    timbre0: torch.Tensor,
    row: dict[str, str],
    enc_device: torch.device,
    kokoro_device: torch.device,
    args: argparse.Namespace,
) -> torch.Tensor | None:
    ph = (row.get("phonemes_st2") or "").strip()
    d_gt = parse_d_gt(row.get("d_gt"))
    if not ph or not d_gt:
        return None
    input_ids, ids, dropped = kokoro_ids(kmodel, ph, kokoro_device)
    if dropped or len(ids) != len(ph):
        return None
    aln = build_alignment(d_gt, kokoro_device)
    if aln.shape[1] != input_ids.shape[1]:
        return None
    try:
        wav = load_wav(row["audio_path"], args.max_wav_sec)
    except Exception:
        return None
    if wav.numel() < SAMPLE_RATE // 4:
        return None
    mel = waveform_to_mel(wav.to(enc_device)).squeeze(0)
    if mel.shape[-1] < 16:
        return None

    timbre = encode_multicrop(
        enc, mel, args.n_crops, args.style_frames, args.min_style_frames
    )
    slot = pack_slot(baseline_pack, len(ph))
    base_p = slot[0, 128:].detach()
    ref_s = make_ref_s(timbre.squeeze(0).to(kokoro_device), base_p)

    with torch.no_grad():
        f0_ov, n_ov = kokoro_f0n_for_ref(kmodel, input_ids, slot, aln)

    audio, _, _, _ = kmodel.forward_with_alignment(
        input_ids, ref_s, aln, f0_override=f0_ov, n_override=n_ov
    )
    if audio.numel() < 100:
        return None
    pred_mel = waveform_to_mel(audio.view(1, -1), device=kokoro_device)
    tgt_mel = waveform_to_mel(wav.unsqueeze(0).to(kokoro_device), device=kokoro_device)
    t = min(pred_mel.shape[-1], tgt_mel.shape[-1])
    if t < 4:
        return None
    loss_mel = F.l1_loss(pred_mel[..., :t], tgt_mel[..., :t].detach())
    loss = args.lambda_mel * loss_mel.to(enc_device)
    if args.lambda_res > 0:
        loss = loss + args.lambda_res * (timbre.squeeze(0) - timbre0).pow(2).mean()
    return loss


@torch.no_grad()
def write_listen(
    out_dir: Path,
    enc: TimbreEncoder,
    kmodel: KModel,
    baseline_pack: torch.Tensor,
    holdout: list[dict[str, str]],
    enc_device: torch.device,
    kokoro_device: torch.device,
    args: argparse.Namespace,
    teacher: torch.Tensor | None,
) -> None:
    listen = out_dir / "listen_test"
    listen.mkdir(parents=True, exist_ok=True)
    enc.eval()
    lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Encoder recon</title>",
        "<style>body{font-family:sans-serif;max-width:900px;margin:1.5rem auto}"
        "audio{width:100%}.card{margin:1rem 0;padding:1rem;border:1px solid #ccc}</style>"
        "</head><body><h1>MEL=MEL TimbreEncoder</h1>",
        "<p>ref | encoder | vector_opt (if any) | baseline</p>",
    ]
    for i, row in enumerate(holdout[: args.listen_n]):
        ph = (row.get("phonemes_st2") or "").strip()
        d_gt = parse_d_gt(row.get("d_gt"))
        if not ph or not d_gt:
            continue
        input_ids, ids, dropped = kokoro_ids(kmodel, ph, kokoro_device)
        if dropped or len(ids) != len(ph):
            continue
        aln = build_alignment(d_gt, kokoro_device)
        if aln.shape[1] != input_ids.shape[1]:
            continue
        wav = load_wav(row["audio_path"], args.max_wav_sec)
        mel = waveform_to_mel(wav.to(enc_device)).squeeze(0)
        timbre = encode_multicrop(
            enc, mel, args.n_crops, args.style_frames, args.min_style_frames
        )
        slot = pack_slot(baseline_pack, len(ph))
        base_p = slot[0, 128:]
        ref_enc = make_ref_s(timbre.squeeze(0).to(kokoro_device), base_p)
        ref_base = slot

        def synth(ref_s: torch.Tensor) -> np.ndarray | None:
            f0, n = kokoro_f0n_for_ref(kmodel, input_ids, slot, aln)
            audio, _, _, _ = kmodel.forward_with_alignment(
                input_ids, ref_s, aln, f0_override=f0, n_override=n
            )
            if audio.numel() < 100:
                return None
            return audio.detach().cpu().numpy().reshape(-1)

        stem = f"{i:02d}_{row['speaker_id']}"
        sf.write(str(listen / f"{stem}_0_ref.wav"), wav.numpy(), SAMPLE_RATE)
        block = [
            f"<div class='card'><h3>{html_lib.escape(stem)}</h3>",
            f"<p>{html_lib.escape((row.get('text') or '')[:160])}</p>",
            f"<p>ref</p><audio controls src='{stem}_0_ref.wav'></audio>",
        ]
        a_e = synth(ref_enc)
        if a_e is not None:
            sf.write(str(listen / f"{stem}_1_encoder.wav"), a_e, SAMPLE_RATE)
            block.append(
                f"<p>encoder</p><audio controls src='{stem}_1_encoder.wav'></audio>"
            )
        if teacher is not None:
            a_t = synth(make_ref_s(teacher.to(kokoro_device), base_p))
            if a_t is not None:
                sf.write(str(listen / f"{stem}_2_vector_opt.wav"), a_t, SAMPLE_RATE)
                block.append(
                    f"<p>vector_opt</p><audio controls src='{stem}_2_vector_opt.wav'></audio>"
                )
        a_b = synth(ref_base)
        if a_b is not None:
            sf.write(str(listen / f"{stem}_3_baseline.wav"), a_b, SAMPLE_RATE)
            block.append(
                f"<p>baseline</p><audio controls src='{stem}_3_baseline.wav'></audio>"
            )
        block.append("</div>")
        lines.extend(block)
        print(f"listen {stem}", flush=True)
    lines.append("</body></html>")
    (listen / "index.html").write_text("\n".join(lines), encoding="utf-8")
    print(f"listen -> {listen / 'index.html'}", flush=True)


def train(args: argparse.Namespace) -> None:
    enc_device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    kokoro_device = torch.device(args.kokoro_device or str(enc_device))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    log_path = out_dir / "train.log"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    rows = load_rows(
        Path(args.manifest),
        args.speaker or None,
        max_clips=args.max_clips,
        max_phonemes=args.max_phonemes,
    )
    holdout, train_rows = split_holdout_alpha(
        rows, holdout_per_spk=args.holdout_per_spk, max_holdout=args.listen_n
    )
    if len(train_rows) < args.batch_size:
        raise SystemExit(
            f"Need >=batch train clips, got {len(train_rows)} (speaker={args.speaker!r})"
        )
    n_spk = len({r["speaker_id"] for r in train_rows})
    steps_per_epoch = max(1, (len(train_rows) + args.batch_size - 1) // args.batch_size)
    total_steps = (
        args.epochs * steps_per_epoch if args.epochs > 0 else args.steps
    )
    log(
        f"enc_device={enc_device} kokoro_device={kokoro_device} "
        f"speaker={args.speaker or 'ALL_OK_ALPHA'} "
        f"train={len(train_rows)} holdout={len(holdout)} speakers={n_spk} "
        f"batch={args.batch_size} n_crops={args.n_crops} lr={args.lr} "
        f"epochs={args.epochs or '-'} steps_per_epoch={steps_per_epoch} "
        f"total_steps={total_steps}"
    )

    pipeline = KPipeline(lang_code="a", model=False)
    need_bank = args.baseline == "mean" or args.voice_bank != "none"
    bank = None
    if need_bank:
        cache = Path(args.bank_cache)
        if not cache.is_file():
            raise SystemExit(f"bank cache missing: {cache}")
        bank = load_bank(cache, device="cpu")
        log(f"loaded bank {cache}")
    if args.baseline == "mean":
        assert bank is not None
        baseline_pack = bank.mean_pack.to(kokoro_device)
        timbre0 = bank.timbre_mean.detach().to(enc_device)
        log("baseline=mean pack; timbre0=bank.timbre_mean")
    else:
        baseline_pack = pipeline.load_voice(args.baseline).to(kokoro_device)
        if baseline_pack.dim() == 2:
            baseline_pack = baseline_pack.unsqueeze(1)
        timbre0 = baseline_pack[0, 0, :128].detach().to(enc_device)
        log(f"baseline={args.baseline}; timbre0=slot0")

    kmodel = KModel(repo_id="hexgrad/Kokoro-82M").to(kokoro_device)
    freeze_module(kmodel)

    enc = TimbreEncoder().to(enc_device)
    enc.train()
    opt = torch.optim.Adam(enc.parameters(), lr=args.lr)

    teacher = None
    if args.teacher_ckpt and Path(args.teacher_ckpt).is_file():
        ck = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
        teacher = (ck["timbre"] if isinstance(ck, dict) else ck).float().view(-1)
        log(f"listen teacher {args.teacher_ckpt}")

    step = 0
    epoch = 0
    while step < total_steps:
        epoch += 1
        order = list(train_rows)
        rng.shuffle(order)
        log(f"=== epoch {epoch}/{args.epochs or '?'} utts={len(order)} ===")
        for i in range(0, len(order), args.batch_size):
            if step >= total_steps:
                break
            batch_rows = order[i : i + args.batch_size]
            opt.zero_grad(set_to_none=True)
            loss_sum = 0.0
            n_ok = 0
            for row in batch_rows:
                try:
                    loss_i = utt_loss(
                        enc,
                        kmodel,
                        baseline_pack,
                        timbre0,
                        row,
                        enc_device,
                        kokoro_device,
                        args,
                    )
                except torch.cuda.OutOfMemoryError:
                    if enc_device.type == "cuda":
                        torch.cuda.empty_cache()
                    log("skip OOM microbatch")
                    continue
                except Exception as e:
                    log(f"skip: {e}")
                    continue
                if loss_i is None:
                    continue
                (loss_i / max(1, len(batch_rows))).backward()
                loss_sum += float(loss_i.detach())
                n_ok += 1
                del loss_i
            if n_ok == 0:
                if enc_device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            torch.nn.utils.clip_grad_norm_(enc.parameters(), args.grad_clip)
            opt.step()
            step += 1
            if step == 1 or step % args.log_every == 0:
                mem = ""
                if enc_device.type == "cuda":
                    alloc = torch.cuda.max_memory_allocated() / (1024**2)
                    reserved = torch.cuda.max_memory_reserved() / (1024**2)
                    mem = f" peak_alloc={alloc:.0f}MiB peak_reserved={reserved:.0f}MiB"
                    torch.cuda.reset_peak_memory_stats()
                log(
                    f"step={step}/{total_steps} epoch={epoch} "
                    f"loss={loss_sum / n_ok:.4f} micro={n_ok}/{len(batch_rows)}{mem}"
                )
            if step % args.save_every == 0:
                path = out_dir / f"encoder_step{step}.pt"
                torch.save(
                    {
                        "step": step,
                        "epoch": epoch,
                        "encoder": enc.state_dict(),
                        "mode": "absolute_recon_mel_eq_mel",
                        "baseline": args.baseline,
                        "args": vars(args),
                    },
                    path,
                )
                log(f"saved {path}")
        epath = out_dir / f"encoder_epoch{epoch}.pt"
        torch.save(
            {
                "step": step,
                "epoch": epoch,
                "encoder": enc.state_dict(),
                "mode": "absolute_recon_mel_eq_mel",
                "baseline": args.baseline,
                "args": vars(args),
            },
            epath,
        )
        log(f"saved {epath}")

    final = out_dir / "encoder_final.pt"
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "encoder": enc.state_dict(),
            "mode": "absolute_recon_mel_eq_mel",
            "baseline": args.baseline,
            "args": vars(args),
        },
        final,
    )
    log(f"saved {final}")
    write_listen(
        out_dir, enc, kmodel, baseline_pack, holdout, enc_device, kokoro_device, args, teacher
    )
    log(f"DONE steps={step} epochs={epoch} -> {final}")


def main() -> None:
    ap = argparse.ArgumentParser(description="TimbreEncoder MEL=MEL reconstruction")
    ap.add_argument(
        "--manifest",
        default=str(ROOT / "timbre_encoder/manifests/train-clean-100.filtered.csv"),
    )
    ap.add_argument(
        "--speaker",
        default="",
        help="Empty = all align_status=ok speakers (alphabetical)",
    )
    ap.add_argument("--baseline", default="mean", help="af_heart | mean | voice name")
    ap.add_argument("--voice-bank", default="us", choices=("none", "us", "core"))
    ap.add_argument(
        "--bank-cache",
        default=str(ROOT / "timbre_encoder/runs/voice_bank_us.pt"),
    )
    ap.add_argument("--epochs", type=int, default=8, help="Full passes over train set")
    ap.add_argument("--steps", type=int, default=0, help="If epochs=0, use fixed steps")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-clips", type=int, default=0, help="0 = all ok utts")
    ap.add_argument("--holdout-per-spk", type=int, default=1)
    ap.add_argument("--max-wav-sec", type=float, default=4.0)
    ap.add_argument(
        "--max-phonemes",
        type=int,
        default=50,
        help="Cap phoneme length (50 fits 4GB all-GPU; 100 spills to Shared)",
    )
    ap.add_argument("--style-frames", type=int, default=192)
    ap.add_argument("--min-style-frames", type=int, default=MIN_STYLE_FRAMES)
    ap.add_argument("--n-crops", type=int, default=4)
    ap.add_argument("--lambda-mel", type=float, default=1.0)
    ap.add_argument(
        "--lambda-res",
        type=float,
        default=0.0,
        help="Unused by default (0). Keep 0 — no ||timbre-timbre0|| term.",
    )
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--listen-n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--teacher-ckpt",
        default="",
        help="Optional vector_opt for listen (empty = skip)",
    )
    ap.add_argument(
        "--out-dir",
        default=str(ROOT / "timbre_encoder/runs/encoder_recon_all_ok"),
    )
    ap.add_argument(
        "--device",
        default="",
        help="TimbreEncoder device (default: cuda if available)",
    )
    ap.add_argument(
        "--kokoro-device",
        default="",
        help="Frozen Kokoro device (empty = same as --device). Use cpu only if needed.",
    )
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
