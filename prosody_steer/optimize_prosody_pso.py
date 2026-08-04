"""
Prosody-only PSO with frozen timbre (E_A / style[:128] never moves).

Like kokoro_hack, but target = original speaker WAV (mel + optional WavLM),
not an emotion centroid. Shared prosody delta[128] on baseline length-slots.

  .venv\\Scripts\\python.exe prosody_steer/optimize_prosody_pso.py \\
      --speaker 6209 \\
      --timbre-ckpt timbre_encoder/runs/speaker_ref_s_6209_bankmean_norm/timbre.pt \\
      --baseline mean --voice-bank us \\
      --max-clips 8 --n-particles 24 --iters 40
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

from kokoro.pipeline import KPipeline  # noqa: E402
from kokoro_revers.model import KModel  # noqa: E402

from timbre_encoder.librispeech_dataset import parse_d_gt  # noqa: E402
from timbre_encoder.mel_preprocess import SAMPLE_RATE, waveform_to_mel  # noqa: E402
from timbre_encoder.optimize_speaker_ref_s import (  # noqa: E402
    assemble_pack_both,
    build_alignment,
    kokoro_ids,
    load_speaker_rows,
    load_timbre_ckpt,
    load_wav,
    make_ref_s,
    pack_slot,
    write_listen,
)
from timbre_encoder.voice_bank import (  # noqa: E402
    CORE_VOICES,
    US_VOICES,
    build_voice_bank,
    load_bank,
    save_bank,
)


@torch.no_grad()
def clip_loss(
    model: KModel,
    row: dict,
    timbre: torch.Tensor,
    prosody_delta: torch.Tensor,
    baseline_pack: torch.Tensor,
    device: torch.device,
    max_wav_sec: float,
    lambda_mel: float,
    lambda_res: float,
    lambda_slm: float,
    slm,
) -> float | None:
    phonemes = (row.get("phonemes_st2") or "").strip()
    d_gt = parse_d_gt(row.get("d_gt"))
    if not phonemes or not d_gt:
        return None

    input_ids, ids, dropped = kokoro_ids(model, phonemes, device)
    if dropped or len(ids) != len(phonemes):
        return None
    aln = build_alignment(d_gt, device)
    if aln.shape[1] != input_ids.shape[1]:
        return None

    try:
        wave = load_wav(row["audio_path"], max_wav_sec).to(device)
    except Exception:
        return None
    if wave.numel() < SAMPLE_RATE // 4:
        return None

    slot = pack_slot(baseline_pack, len(phonemes))
    base_p = slot[0, 128:].detach()
    ref_s = make_ref_s(timbre, base_p + prosody_delta)

    audio_pred, _, _, _ = model.forward_with_alignment(input_ids, ref_s, aln)
    if audio_pred.dim() == 0 or audio_pred.numel() < 100:
        return None

    loss = 0.0
    if lambda_mel > 0:
        pred_mel = waveform_to_mel(
            audio_pred.unsqueeze(0) if audio_pred.dim() == 1 else audio_pred,
            device=device,
        )
        tgt_mel = waveform_to_mel(wave.unsqueeze(0), device=device)
        t = min(pred_mel.shape[-1], tgt_mel.shape[-1])
        if t < 4:
            return None
        loss += lambda_mel * float(F.l1_loss(pred_mel[..., :t], tgt_mel[..., :t]).item())

    if lambda_res > 0:
        loss += lambda_res * float(prosody_delta.pow(2).mean().item())

    if slm is not None and lambda_slm > 0:
        pred_w = audio_pred.view(1, -1) if audio_pred.dim() == 1 else audio_pred
        real_w = wave.view(1, -1)
        tw = min(pred_w.shape[-1], real_w.shape[-1])
        loss += lambda_slm * float(slm(pred_w[..., :tw], real_w[..., :tw]).item())

    return loss


@torch.no_grad()
def batch_loss(
    model: KModel,
    rows: list[dict],
    timbre: torch.Tensor,
    prosody_delta: torch.Tensor,
    baseline_pack: torch.Tensor,
    device: torch.device,
    max_wav_sec: float,
    lambda_mel: float,
    lambda_res: float,
    lambda_slm: float,
    slm,
    eval_clips: int,
    rng: random.Random,
) -> float:
    if not rows:
        return float("inf")
    sample = rows if eval_clips <= 0 or eval_clips >= len(rows) else rng.sample(rows, eval_clips)
    vals: list[float] = []
    for row in sample:
        v = clip_loss(
            model,
            row,
            timbre,
            prosody_delta,
            baseline_pack,
            device,
            max_wav_sec,
            lambda_mel,
            lambda_res,
            lambda_slm,
            slm,
        )
        if v is not None:
            vals.append(v)
    return sum(vals) / len(vals) if vals else float("inf")


def main() -> None:
    ap = argparse.ArgumentParser(description="PSO prosody delta; frozen timbre (E_A)")
    ap.add_argument("--speaker", type=str, default="6209")
    ap.add_argument(
        "--manifest",
        type=str,
        default=str(ROOT / "timbre_encoder/manifests/train-clean-100.filtered.csv"),
    )
    ap.add_argument("--baseline", type=str, default="mean")
    ap.add_argument("--voice-bank", choices=("none", "us", "core"), default="us")
    ap.add_argument(
        "--timbre-ckpt",
        type=str,
        default=str(ROOT / "timbre_encoder/runs/speaker_ref_s_6209_bankmean_norm/timbre.pt"),
    )
    ap.add_argument("--max-clips", type=int, default=8)
    ap.add_argument("--max-phonemes", type=int, default=80)
    ap.add_argument("--max-wav-sec", type=float, default=6.0)
    ap.add_argument("--holdout", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-particles", type=int, default=24)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument(
        "--eval-clips",
        type=int,
        default=0,
        help="Clips per particle eval (0=all train clips; prefer 0 when max-clips is small)",
    )
    ap.add_argument("--pos-min", type=float, default=-1.5)
    ap.add_argument("--pos-max", type=float, default=1.5)
    ap.add_argument("--w", type=float, default=0.72)
    ap.add_argument("--c1", type=float, default=1.49)
    ap.add_argument("--c2", type=float, default=1.49)
    ap.add_argument("--vel-clip", type=float, default=0.35)
    ap.add_argument("--early-stopping", type=int, default=12)
    ap.add_argument("--lambda-mel", type=float, default=1.0)
    ap.add_argument("--lambda-res", type=float, default=0.3)
    ap.add_argument("--lambda-slm", type=float, default=0.0)
    ap.add_argument("--slm-device", type=str, default="cpu")
    ap.add_argument("--slm-layers", type=str, default="4,8,12")
    ap.add_argument("--bank-cache", type=str, default="")
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (
        HERE / "runs" / f"speaker_prosody_pso_{args.speaker}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} speaker={args.speaker} frozen_timbre={args.timbre_ckpt} "
        f"baseline={args.baseline} clips={args.max_clips} particles={args.n_particles} "
        f"iters={args.iters} out={out_dir}",
        flush=True,
    )

    slm = None
    if args.lambda_slm > 0:
        from timbre_encoder.slm_loss import WavLMFeatureLoss

        layers = [int(x) for x in args.slm_layers.split(",") if x.strip()]
        print(f"Loading WavLM on {args.slm_device} layers={layers}...", flush=True)
        slm = WavLMFeatureLoss(layers=layers, device=args.slm_device)

    rows = load_speaker_rows(
        Path(args.manifest),
        args.speaker,
        max_phonemes=args.max_phonemes,
        max_clips=args.max_clips + args.holdout,
        seed=args.seed,
    )
    if len(rows) < args.holdout + 1:
        raise SystemExit(
            f"Need at least {args.holdout + 1} clips for speaker {args.speaker}, got {len(rows)}"
        )
    holdout = rows[: args.holdout]
    train_rows = rows[args.holdout :]
    if args.max_clips > 0:
        train_rows = train_rows[: args.max_clips]
    print(f"train_clips={len(train_rows)} holdout={len(holdout)}", flush=True)

    pipeline = KPipeline(lang_code="a", model=False)
    bank = None
    if args.voice_bank != "none":
        cache = Path(args.bank_cache) if args.bank_cache else (out_dir / "voice_bank.pt")
        if cache.is_file():
            bank = load_bank(cache, device=device)
        else:
            names = US_VOICES if args.voice_bank == "us" else CORE_VOICES
            bank = build_voice_bank(pipeline, names).to(device)
            cache.parent.mkdir(parents=True, exist_ok=True)
            save_bank(bank, cache)

    if args.baseline == "mean":
        if bank is None:
            raise SystemExit("--baseline mean requires --voice-bank us|core")
        baseline_pack = bank.mean_pack.to(device)
    else:
        baseline_pack = pipeline.load_voice(args.baseline).to(device)
        if baseline_pack.dim() == 2:
            baseline_pack = baseline_pack.unsqueeze(1)

    # Timbre frozen — never updated
    timbre = load_timbre_ckpt(Path(args.timbre_ckpt), device).detach()
    print(f"timbre frozen |t|={timbre.norm().item():.4f}", flush=True)

    model = KModel(repo_id="hexgrad/Kokoro-82M").to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    dim = 128
    g = torch.Generator(device=device)
    g.manual_seed(args.seed)
    rng = random.Random(args.seed)

    pos = args.pos_min + (args.pos_max - args.pos_min) * torch.rand(
        args.n_particles, dim, device=device, generator=g
    )
    # Particle 0 starts at zero delta (baseline prosody)
    pos[0].zero_()
    vel = torch.zeros_like(pos)
    pbest_pos = pos.clone()
    pbest_loss = torch.full((args.n_particles,), float("inf"), device=device)
    gbest_pos = torch.zeros(dim, device=device)
    gbest_loss = float("inf")
    no_improve = 0

    zero_loss = batch_loss(
        model,
        train_rows,
        timbre,
        torch.zeros(dim, device=device),
        baseline_pack,
        device,
        args.max_wav_sec,
        args.lambda_mel,
        args.lambda_res,
        args.lambda_slm,
        slm,
        eval_clips=0,
        rng=rng,
    )
    print(f"[PSO] zero-delta loss={zero_loss:.6f}", flush=True)

    history: list[dict] = []
    try:
        for it in range(args.iters):
            losses = []
            for p in range(args.n_particles):
                l = batch_loss(
                    model,
                    train_rows,
                    timbre,
                    pos[p],
                    baseline_pack,
                    device,
                    args.max_wav_sec,
                    args.lambda_mel,
                    args.lambda_res,
                    args.lambda_slm,
                    slm,
                    eval_clips=args.eval_clips,
                    rng=rng,
                )
                losses.append(l)

            losses_t = torch.tensor(losses, device=device)
            improved = losses_t < pbest_loss
            pbest_loss = torch.where(improved, losses_t, pbest_loss)
            pbest_pos[improved] = pos[improved]

            min_idx = int(torch.argmin(pbest_loss).item())
            current = float(pbest_loss[min_idx].item())
            if current < gbest_loss:
                gbest_loss = current
                gbest_pos = pbest_pos[min_idx].clone()
                no_improve = 0
            else:
                no_improve += 1

            hist = {"iter": it, "gbest_loss": gbest_loss, "mean_pbest": float(pbest_loss.mean())}
            history.append(hist)
            print(
                f"[PSO] iter={it:03d} gbest={gbest_loss:.6f} "
                f"mean_pbest={hist['mean_pbest']:.6f} no_improve={no_improve}",
                flush=True,
            )

            if no_improve >= args.early_stopping:
                print(f"[PSO] early stop at iter={it}", flush=True)
                break

            r1 = torch.rand(args.n_particles, dim, device=device, generator=g)
            r2 = torch.rand(args.n_particles, dim, device=device, generator=g)
            vel = (
                args.w * vel
                + args.c1 * r1 * (pbest_pos - pos)
                + args.c2 * r2 * (gbest_pos.unsqueeze(0) - pos)
            )
            vel = torch.clamp(vel, -args.vel_clip, args.vel_clip)
            pos = torch.clamp(pos + vel, args.pos_min, args.pos_max)
    except KeyboardInterrupt:
        print("\n[PSO] interrupted; saving best so far", flush=True)

    # Full-batch score for best
    final_loss = batch_loss(
        model,
        train_rows,
        timbre,
        gbest_pos,
        baseline_pack,
        device,
        args.max_wav_sec,
        args.lambda_mel,
        args.lambda_res,
        args.lambda_slm,
        slm,
        eval_clips=0,
        rng=rng,
    )
    print(
        f"[PSO] done zero={zero_loss:.6f} gbest_eval={gbest_loss:.6f} "
        f"gbest_full={final_loss:.6f} |delta|={gbest_pos.norm().item():.4f}",
        flush=True,
    )

    timbre_cpu = timbre.detach().cpu()
    delta_cpu = gbest_pos.detach().cpu()
    pack = assemble_pack_both(timbre_cpu, delta_cpu, baseline_pack.cpu(), mode="delta")
    torch.save(timbre_cpu, out_dir / "timbre.pt")
    torch.save(delta_cpu, out_dir / "prosody_delta.pt")
    torch.save(pack, out_dir / "voice_pack.pt")
    torch.save(
        {
            "timbre": timbre_cpu,
            "prosody_delta": delta_cpu,
            "prosody_mode": "shared",
            "speaker": args.speaker,
            "baseline": args.baseline,
            "method": "pso",
            "zero_loss": zero_loss,
            "gbest_loss": gbest_loss,
            "gbest_full_loss": final_loss,
            "history": history,
            "args": vars(args),
        },
        out_dir / "style.pt",
    )
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "speaker": args.speaker,
                "method": "pso",
                "timbre_ckpt": args.timbre_ckpt,
                "zero_loss": zero_loss,
                "gbest_loss": gbest_loss,
                "gbest_full_loss": final_loss,
                "delta_norm": float(delta_cpu.norm()),
                "train_clips": len(train_rows),
                "iters_ran": len(history),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    write_listen(
        out_dir,
        model,
        baseline_pack,
        timbre,
        gbest_pos,
        holdout,
        device,
        args.max_wav_sec,
        title=f"Speaker {args.speaker} prosody PSO frozen E_A base={args.baseline}",
        prosody_mode="shared",
        slm_scorer=slm,
    )
    print(f"Saved -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
