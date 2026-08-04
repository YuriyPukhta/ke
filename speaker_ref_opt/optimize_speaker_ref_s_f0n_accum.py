"""
Optimize Kokoro style: mel + cached F0/N + soft duration loss, with grad-accum.

Decoder / listen always use forced GT alignment (d_gt).
Duration loss compares soft duration_proj prediction vs d_gt (aux for prosody);
it does NOT drive synthesis timing.

  .venv\\Scripts\\python.exe speaker_ref_opt/optimize_speaker_ref_s_f0n_accum.py \\
      --speaker 6209 --baseline mean --voice-bank us --lr 3e-3 \\
      --steps 200 --batch-size 8 --lambda-dur 1.0
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))
sys.path.insert(0, str(HERE))

from kokoro.pipeline import KPipeline  # noqa: E402
from kokoro_revers.model import KModel  # noqa: E402

from gt_cache import DEFAULT_FEATURES_ROOT, feature_path, load_curve  # noqa: E402
from timbre_encoder.kokoro_embed_train.common import (  # noqa: E402
    calibrate_st2_to_kokoro,
    d_gt_target_with_bos_eos,
    kokoro_f0n_for_ref,
    prosody_heads_forward_grad,
    to_1d,
)
from timbre_encoder.librispeech_dataset import parse_d_gt  # noqa: E402
from timbre_encoder.mel_preprocess import SAMPLE_RATE, waveform_to_mel  # noqa: E402
from timbre_encoder.voice_bank import (  # noqa: E402
    CORE_VOICES,
    US_VOICES,
    build_voice_bank,
    load_bank,
    normalize_to_bank,
    save_bank,
)
from timbre_encoder.voice_pack import assemble_voice_pack  # noqa: E402

from optimize_speaker_ref_s import (  # noqa: E402
    assemble_pack_both,
    build_alignment,
    kokoro_ids,
    load_timbre_ckpt,
    load_wav,
    make_ref_s,
    pack_slot,
    write_listen,
)


def load_speaker_rows_f0n(
    csv_path: Path,
    speaker: str,
    max_phonemes: int,
    max_clips: int,
    seed: int,
    features_root: Path,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("speaker_id") != speaker:
                continue
            if row.get("align_status") != "ok":
                continue
            if row.get("f0_status") != "ok" or row.get("n_status") != "ok":
                continue
            ph_st2 = (row.get("phonemes_st2") or "").strip()
            d_gt = parse_d_gt(row.get("d_gt"))
            if not ph_st2 or not d_gt:
                continue
            if len(d_gt) != len(ph_st2):
                continue
            if not (2 <= len(ph_st2) <= max_phonemes):
                continue
            f0_p = feature_path(row["audio_path"], "f0", features_root)
            n_p = feature_path(row["audio_path"], "n", features_root)
            if not f0_p.is_file() or not n_p.is_file():
                continue
            rows.append(row)
    rng = random.Random(seed)
    rng.shuffle(rows)
    if max_clips > 0 and len(rows) > max_clips:
        rows = rows[:max_clips]
    return rows


def load_cached_f0n(
    audio_path: str,
    t_mel: int,
    features_root: Path,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    f0 = load_curve(feature_path(audio_path, "f0", features_root))
    n = load_curve(feature_path(audio_path, "n", features_root))
    if f0 is None or n is None:
        return None
    if f0.numel() < t_mel or n.numel() < t_mel:
        return None
    return f0[:t_mel], n[:t_mel]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Optimize style with mel + cached F0/N + dur, grad-accum"
    )
    ap.add_argument("--speaker", type=str, default="6209")
    ap.add_argument(
        "--manifest",
        type=str,
        default=str(ROOT / "timbre_encoder/manifests/train-clean-100.filtered.csv"),
    )
    ap.add_argument(
        "--features-root",
        type=str,
        default=str(DEFAULT_FEATURES_ROOT),
        help="Root of gt_cache/features (f0/ and n/)",
    )
    ap.add_argument("--baseline", type=str, default="mean")
    ap.add_argument(
        "--train-target",
        choices=("timbre", "prosody", "both"),
        default="both",
        help="both=train [:128]+[128:] (needed for F0/N/dur grads into prosody)",
    )
    ap.add_argument(
        "--prosody-mode",
        choices=("shared", "slot", "absolute"),
        default="absolute",
    )
    ap.add_argument("--voice-bank", choices=("none", "us", "core"), default="us")
    ap.add_argument("--norm-to-bank", action="store_true")
    ap.add_argument("--z-clip", type=float, default=2.5)
    ap.add_argument("--bank-cache", type=str, default="")
    ap.add_argument(
        "--timbre-ckpt",
        type=str,
        default="",
        help="Init/freeze timbre .pt (required for train-target=prosody)",
    )
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Gradient accumulation: average N clips per optimizer step",
    )
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lambda-mel", type=float, default=1.0)
    ap.add_argument("--lambda-res", type=float, default=0.1)
    ap.add_argument("--lambda-smooth", type=float, default=0.0)
    ap.add_argument("--lambda-f0", type=float, default=1.0)
    ap.add_argument("--lambda-n", type=float, default=1.0)
    ap.add_argument(
        "--lambda-dur",
        type=float,
        default=1.0,
        help="Soft duration_proj vs d_gt (aux); decode still uses forced d_gt",
    )
    ap.add_argument("--lambda-slm", type=float, default=0.0)
    ap.add_argument("--slm-device", type=str, default="cpu")
    ap.add_argument("--slm-layers", type=str, default="4,8,12")
    ap.add_argument(
        "--slm-cos",
        action="store_true",
        help="WavLM loss = L1 + (1-cos) mean-pooled (default off; L1 only)",
    )
    ap.add_argument("--max-clips", type=int, default=64)
    ap.add_argument("--max-phonemes", type=int, default=100)
    ap.add_argument(
        "--max-wav-sec",
        type=float,
        default=8.0,
        help="Match f0n_cache default (8s); curves truncated to mel frames",
    )
    ap.add_argument("--holdout", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="")
    ap.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Use raw ST2 F0/N with simple scale match (legacy); default=calibrate to baseline",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (
        HERE / "runs" / f"speaker_ref_s_f0n_accum_{args.speaker}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    features_root = Path(args.features_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    print(
        f"device={device} speaker={args.speaker} target={args.train_target} "
        f"prosody_mode={args.prosody_mode} baseline={args.baseline} "
        f"batch={args.batch_size} lr={args.lr} mel={args.lambda_mel} res={args.lambda_res} "
        f"f0={args.lambda_f0} n={args.lambda_n} dur={args.lambda_dur} "
        f"slm={args.lambda_slm} slm_cos={args.slm_cos} "
        f"calibrate={not args.no_calibrate} features={features_root} out={out_dir}",
        flush=True,
    )

    slm = None
    if args.lambda_slm > 0:
        from timbre_encoder.slm_loss import WavLMFeatureLoss

        layers = [int(x) for x in args.slm_layers.split(",") if x.strip()]
        print(
            f"Loading WavLM on {args.slm_device} layers={layers} "
            f"include_cos={args.slm_cos}...",
            flush=True,
        )
        slm = WavLMFeatureLoss(
            layers=layers, device=args.slm_device, include_cos=args.slm_cos
        )

    rows = load_speaker_rows_f0n(
        Path(args.manifest),
        args.speaker,
        max_phonemes=args.max_phonemes,
        max_clips=args.max_clips + args.holdout,
        seed=args.seed,
        features_root=features_root,
    )
    if len(rows) < args.holdout + 1:
        raise SystemExit(
            f"Need >= {args.holdout + 1} clips with align+f0+n cache for "
            f"speaker {args.speaker}, got {len(rows)}"
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
            print(f"Loading voice bank cache {cache}", flush=True)
            bank = load_bank(cache, device=device)
        else:
            names = US_VOICES if args.voice_bank == "us" else CORE_VOICES
            bank = build_voice_bank(pipeline, names).to(device)
            cache.parent.mkdir(parents=True, exist_ok=True)
            save_bank(bank, cache)
            print(f"Saved voice bank -> {cache}", flush=True)

    if args.baseline == "mean":
        if bank is None:
            raise SystemExit("--baseline mean requires --voice-bank us|core")
        baseline_pack = bank.mean_pack.to(device)
        print("baseline_pack = voice-bank MEAN pack", flush=True)
    else:
        baseline_pack = pipeline.load_voice(args.baseline).to(device)
        if baseline_pack.dim() == 2:
            baseline_pack = baseline_pack.unsqueeze(1)

    model = KModel(repo_id="hexgrad/Kokoro-82M").to(device)
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)

    prosody0: torch.Tensor | None = None
    if args.train_target in ("timbre", "both"):
        if args.timbre_ckpt:
            timbre0 = load_timbre_ckpt(Path(args.timbre_ckpt), device).detach()
            print(f"init timbre from {args.timbre_ckpt}", flush=True)
        elif bank is not None:
            timbre0 = bank.timbre_mean.detach().clone()
            print("init timbre = bank.timbre_mean", flush=True)
        else:
            timbre0 = baseline_pack[0, 0, :128].detach().clone()
        timbre = timbre0.clone().requires_grad_(True)
    else:
        if not args.timbre_ckpt:
            raise SystemExit("--train-target prosody requires --timbre-ckpt")
        timbre = load_timbre_ckpt(Path(args.timbre_ckpt), device).detach()
        if args.norm_to_bank and bank is not None:
            timbre = normalize_to_bank(timbre, bank.timbre_mean, bank.timbre_std, args.z_clip)
        timbre0 = timbre.detach().clone()
        print(f"frozen timbre from {args.timbre_ckpt}", flush=True)

    if args.train_target in ("prosody", "both"):
        n_slots = baseline_pack.shape[0]
        if args.prosody_mode == "absolute":
            if bank is not None:
                prosody0 = bank.prosody_mean.detach().clone()
                print("init absolute prosody = bank.prosody_mean", flush=True)
            else:
                prosody0 = baseline_pack[0, 0, 128:].detach().clone()
                print("init absolute prosody = baseline slot0", flush=True)
            prosody_delta = prosody0.clone().requires_grad_(True)
        elif args.prosody_mode == "slot":
            prosody_delta = torch.zeros(n_slots, 128, device=device, requires_grad=True)
            print(f"train per-slot prosody delta [{n_slots},128]", flush=True)
        else:
            prosody_delta = torch.zeros(128, device=device, requires_grad=True)
            print("train shared prosody delta", flush=True)
    else:
        prosody_delta = None

    trainable = []
    if args.train_target in ("timbre", "both"):
        trainable.append(timbre)
    if args.train_target in ("prosody", "both"):
        assert prosody_delta is not None
        trainable.append(prosody_delta)
    opt = torch.optim.Adam(trainable, lr=args.lr)
    rng = random.Random(args.seed)
    history: list[dict[str, float]] = []
    skipped = 0
    bs = args.batch_size

    def microbatch_loss(row: dict[str, str]) -> dict[str, torch.Tensor] | None:
        """One clip forward; returns loss components or None to skip."""
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
            wave = load_wav(row["audio_path"], args.max_wav_sec).to(device)
        except Exception:
            return None
        if wave.numel() < SAMPLE_RATE // 4:
            return None

        tgt_mel = waveform_to_mel(wave.unsqueeze(0), device=device)
        t_mel = int(tgt_mel.shape[-1])
        cached = load_cached_f0n(row["audio_path"], t_mel, features_root)
        if cached is None:
            return None
        f0_st2, n_st2 = cached

        slot = pack_slot(baseline_pack, len(phonemes))
        idx = max(0, min(len(phonemes) - 1, baseline_pack.shape[0] - 1))
        base_p = slot[0, 128:].detach()

        if args.train_target == "timbre":
            ref_s = make_ref_s(timbre, base_p)
            loss_res = (timbre - timbre0).pow(2).mean()
            loss_smooth = ref_s.new_zeros(())
        else:
            assert prosody_delta is not None
            if args.prosody_mode == "absolute":
                ref_s = make_ref_s(timbre, prosody_delta)
                assert prosody0 is not None
                loss_res = (prosody_delta - prosody0).pow(2).mean()
                if args.train_target == "both":
                    loss_res = loss_res + (timbre - timbre0).pow(2).mean()
                loss_smooth = ref_s.new_zeros(())
            elif args.prosody_mode == "slot":
                d_slot = prosody_delta[idx]
                ref_s = make_ref_s(timbre, base_p + d_slot)
                loss_res = d_slot.pow(2).mean()
                if args.train_target == "both":
                    loss_res = loss_res + (timbre - timbre0).pow(2).mean()
                loss_smooth = ref_s.new_zeros(())
                if args.lambda_smooth > 0 and idx + 1 < prosody_delta.shape[0]:
                    loss_smooth = loss_smooth + (
                        prosody_delta[idx] - prosody_delta[idx + 1]
                    ).pow(2).mean()
                if args.lambda_smooth > 0 and idx > 0:
                    loss_smooth = loss_smooth + (
                        prosody_delta[idx] - prosody_delta[idx - 1]
                    ).pow(2).mean()
            else:
                ref_s = make_ref_s(timbre, base_p + prosody_delta)
                loss_res = prosody_delta.pow(2).mean()
                if args.train_target == "both":
                    loss_res = loss_res + (timbre - timbre0).pow(2).mean()
                loss_smooth = ref_s.new_zeros(())

        # Soft duration (aux): duration_proj vs d_gt. Decode still uses forced aln.
        loss_dur = ref_s.new_zeros(())
        if args.lambda_dur > 0:
            soft_dur, _, _ = prosody_heads_forward_grad(model, input_ids, ref_s, aln)
            dur_tgt = d_gt_target_with_bos_eos(d_gt, device)
            if soft_dur.numel() == dur_tgt.numel():
                loss_dur = F.l1_loss(soft_dur, dur_tgt)

        # Decoder: forced GT alignment → mel / F0 / N from current ref_s
        audio_pred, f0_pred, n_pred, _ = model.forward_with_alignment(input_ids, ref_s, aln)
        if audio_pred.dim() == 0 or audio_pred.numel() < 100:
            return None

        pred_mel = waveform_to_mel(
            audio_pred.unsqueeze(0) if audio_pred.dim() == 1 else audio_pred,
            device=device,
        )
        t = min(pred_mel.shape[-1], tgt_mel.shape[-1])
        if t < 4:
            return None
        if args.lambda_mel > 0:
            loss_mel = F.l1_loss(pred_mel[..., :t], tgt_mel[..., :t].detach())
        else:
            loss_mel = ref_s.new_zeros(())

        loss_slm = ref_s.new_zeros(())
        if slm is not None and args.lambda_slm > 0:
            pred_w = audio_pred.view(1, -1) if audio_pred.dim() == 1 else audio_pred
            real_w = wave.view(1, -1)
            tw = min(pred_w.shape[-1], real_w.shape[-1])
            loss_slm = slm(pred_w[..., :tw], real_w[..., :tw].detach())

        loss_f0 = ref_s.new_zeros(())
        loss_n = ref_s.new_zeros(())
        if args.lambda_f0 > 0 or args.lambda_n > 0:
            f0_p = to_1d(f0_pred)
            n_p = to_1d(n_pred)
            if args.no_calibrate:
                f0_tgt = f0_st2.to(device)
                e_tgt = n_st2.to(device)
                if args.lambda_f0 > 0 and f0_p.numel() > 4 and f0_tgt.numel() > 4:
                    f0_pi = F.interpolate(
                        f0_p.view(1, 1, -1), size=f0_tgt.numel(), mode="linear", align_corners=False
                    ).view(-1)
                    f0_tgt_n = f0_tgt / 500.0
                    scale = f0_tgt_n.mean() / (f0_pi.detach().abs().mean() + 1e-3)
                    loss_f0 = F.l1_loss(f0_pi * scale, f0_tgt_n)
                if args.lambda_n > 0 and n_p.numel() > 4 and e_tgt.numel() > 4:
                    n_pi = F.interpolate(
                        n_p.view(1, 1, -1), size=e_tgt.numel(), mode="linear", align_corners=False
                    ).view(-1)
                    e_n = (e_tgt - e_tgt.mean()) / (e_tgt.std() + 1e-3)
                    n_n = (n_pi - n_pi.mean()) / (n_pi.std() + 1e-3)
                    loss_n = F.l1_loss(n_n, e_n)
            else:
                f0_ref, n_ref = kokoro_f0n_for_ref(model, input_ids, slot.detach(), aln)
                f0_cal, n_cal = calibrate_st2_to_kokoro(
                    f0_st2.to(device), n_st2.to(device), f0_ref, n_ref
                )
                if args.lambda_f0 > 0 and f0_p.numel() > 4 and f0_cal.numel() > 4:
                    tf = min(f0_p.numel(), f0_cal.numel())
                    loss_f0 = F.l1_loss(f0_p[:tf], f0_cal[:tf].detach())
                if args.lambda_n > 0 and n_p.numel() > 4 and n_cal.numel() > 4:
                    tn = min(n_p.numel(), n_cal.numel())
                    loss_n = F.l1_loss(n_p[:tn], n_cal[:tn].detach())

        loss = (
            args.lambda_mel * loss_mel
            + args.lambda_res * loss_res
            + args.lambda_smooth * loss_smooth
            + args.lambda_f0 * loss_f0
            + args.lambda_n * loss_n
            + args.lambda_dur * loss_dur
            + args.lambda_slm * loss_slm
        )
        return {
            "loss": loss,
            "mel": loss_mel,
            "res": loss_res,
            "f0": loss_f0,
            "n": loss_n,
            "dur": loss_dur,
            "slm": loss_slm,
            "slot_idx": torch.tensor(float(idx)),
        }

    for step in range(1, args.steps + 1):
        opt.zero_grad()
        acc = {
            "loss": 0.0,
            "mel": 0.0,
            "res": 0.0,
            "f0": 0.0,
            "n": 0.0,
            "dur": 0.0,
            "slm": 0.0,
        }
        n_ok = 0
        attempts = 0
        max_attempts = bs * 20
        last_idx = 0
        while n_ok < bs and attempts < max_attempts:
            attempts += 1
            row = train_rows[rng.randrange(len(train_rows))]
            parts = microbatch_loss(row)
            if parts is None:
                skipped += 1
                continue
            # Mean over batch: each micro contributes 1/bs
            (parts["loss"] / bs).backward()
            for k in acc:
                acc[k] += float(parts[k].detach().item())
            last_idx = int(parts["slot_idx"].item())
            n_ok += 1

        if n_ok == 0:
            print(f"step={step} WARN: empty batch, skipping update", flush=True)
            continue

        # If fewer than bs succeeded, scale grads so mean stays correct
        if n_ok < bs:
            scale = bs / n_ok
            for p in trainable:
                if p.grad is not None:
                    p.grad.mul_(scale)

        opt.step()

        if args.norm_to_bank and bank is not None:
            with torch.no_grad():
                if args.train_target in ("timbre", "both") and timbre.requires_grad:
                    timbre.copy_(
                        normalize_to_bank(timbre, bank.timbre_mean, bank.timbre_std, args.z_clip)
                    )
                if prosody_delta is not None and prosody_delta.requires_grad:
                    if args.prosody_mode == "absolute":
                        prosody_delta.copy_(
                            normalize_to_bank(
                                prosody_delta, bank.prosody_mean, bank.prosody_std, args.z_clip
                            )
                        )
                    elif args.prosody_mode == "slot":
                        base_p = pack_slot(baseline_pack, last_idx + 1)[0, 128:].detach()
                        abs_p = base_p + prosody_delta[last_idx]
                        abs_n = normalize_to_bank(
                            abs_p, bank.prosody_mean, bank.prosody_std, args.z_clip
                        )
                        prosody_delta[last_idx].copy_(abs_n - base_p)
                    else:
                        base_p = pack_slot(baseline_pack, last_idx + 1)[0, 128:].detach()
                        abs_p = base_p + prosody_delta
                        abs_n = normalize_to_bank(
                            abs_p, bank.prosody_mean, bank.prosody_std, args.z_clip
                        )
                        prosody_delta.copy_(abs_n - base_p)

        hist = {
            "step": float(step),
            "loss": acc["loss"] / n_ok,
            "mel": acc["mel"] / n_ok,
            "res": acc["res"] / n_ok,
            "f0": acc["f0"] / n_ok,
            "n": acc["n"] / n_ok,
            "dur": acc["dur"] / n_ok,
            "slm": acc["slm"] / n_ok,
            "batch_ok": float(n_ok),
        }
        history.append(hist)
        if step == 1 or step % 10 == 0 or step == args.steps:
            print(
                f"step={step}/{args.steps} batch={n_ok}/{bs} "
                f"loss={hist['loss']:.4f} mel={hist['mel']:.4f} "
                f"f0={hist['f0']:.4f} n={hist['n']:.4f} dur={hist['dur']:.4f} "
                f"res={hist['res']:.4f} skipped={skipped}",
                flush=True,
            )

    timbre_cpu = timbre.detach().cpu()
    delta_cpu = prosody_delta.detach().cpu() if prosody_delta is not None else None
    ckpt = {
        "timbre": timbre_cpu,
        "prosody_delta": delta_cpu,
        "prosody_mode": args.prosody_mode if args.train_target != "timbre" else None,
        "speaker": args.speaker,
        "baseline": args.baseline,
        "train_target": args.train_target,
        "f0n_source": "st2_cache",
        "calibrate": not args.no_calibrate,
        "batch_size": args.batch_size,
        "lambda_dur": args.lambda_dur,
    }
    torch.save(ckpt, out_dir / "style.pt")
    torch.save({"timbre": timbre_cpu, "speaker": args.speaker, "baseline": args.baseline}, out_dir / "timbre.pt")
    if delta_cpu is not None:
        torch.save(delta_cpu, out_dir / "prosody_delta.pt")
        if args.prosody_mode == "absolute":
            torch.save(
                {"prosody": delta_cpu, "speaker": args.speaker, "baseline": args.baseline},
                out_dir / "prosody.pt",
            )
            pack = assemble_pack_both(timbre_cpu, delta_cpu, baseline_pack.cpu(), mode="replace")
        else:
            pack_mode = "slot_delta" if args.prosody_mode == "slot" else "delta"
            pack = assemble_pack_both(timbre_cpu, delta_cpu, baseline_pack.cpu(), mode=pack_mode)
    else:
        pack = assemble_voice_pack(timbre_cpu, baseline_pack.cpu())
    torch.save(pack, out_dir / "voice_pack.pt")

    first_n = history[:50]
    last_n = history[-50:] if len(history) >= 50 else history

    def avg(key: str, xs: list[dict[str, float]]) -> float:
        if not xs:
            return float("nan")
        return sum(h[key] for h in xs) / len(xs)

    if args.train_target == "timbre":
        vec_l2 = float((timbre_cpu - timbre0.cpu()).pow(2).sum().sqrt())
        touched = 0
    else:
        assert delta_cpu is not None
        if args.prosody_mode == "absolute":
            touched = 1
            assert prosody0 is not None
            vec_l2 = float((delta_cpu - prosody0.detach().cpu()).pow(2).sum().sqrt())
            if args.train_target == "both":
                vec_l2 = float(
                    (timbre_cpu - timbre0.cpu()).pow(2).sum().sqrt()
                    + (delta_cpu - prosody0.detach().cpu()).pow(2).sum().sqrt()
                )
        elif delta_cpu.dim() == 2:
            slot_norm = delta_cpu.pow(2).sum(dim=-1).sqrt()
            touched = int((slot_norm > 1e-4).sum().item())
            vec_l2 = float(slot_norm.mean())
        else:
            touched = 1
            vec_l2 = float(delta_cpu.pow(2).sum().sqrt())

    summary = {
        "speaker": args.speaker,
        "train_target": args.train_target,
        "prosody_mode": args.prosody_mode,
        "baseline": args.baseline,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lambda_f0": args.lambda_f0,
        "lambda_n": args.lambda_n,
        "lambda_dur": args.lambda_dur,
        "f0n_calibrate": not args.no_calibrate,
        "train_clips": len(train_rows),
        "skipped": skipped,
        "mel_first50": avg("mel", first_n),
        "mel_last50": avg("mel", last_n),
        "f0_last50": avg("f0", last_n),
        "n_last50": avg("n", last_n),
        "dur_last50": avg("dur", last_n),
        "res_last50": avg("res", last_n),
        "vec_l2_from_start": vec_l2,
        "slots_touched": touched,
    }
    print("summary", summary, flush=True)
    (out_dir / "summary.txt").write_text(
        "\n".join(f"{k}={v}" for k, v in summary.items()) + "\n",
        encoding="utf-8",
    )

    model.eval()
    from timbre_encoder.slm_loss import WavLMFeatureLoss

    print("Loading WavLM scorer (CPU) for listen page...", flush=True)
    slm_scorer = WavLMFeatureLoss(layers=[6, 12], device="cpu")
    write_listen(
        out_dir,
        model,
        baseline_pack,
        timbre,
        prosody_delta,
        holdout,
        device,
        args.max_wav_sec,
        title=(
            f"Speaker {args.speaker} {args.train_target}+F0N+dur "
            f"bs={args.batch_size} mode={args.prosody_mode} "
            f"base={args.baseline} lr={args.lr}"
        ),
        prosody_mode=args.prosody_mode if args.train_target != "timbre" else "shared",
        slm_scorer=slm_scorer,
    )
    print(f"saved {out_dir} listen={out_dir / 'listen_test' / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
