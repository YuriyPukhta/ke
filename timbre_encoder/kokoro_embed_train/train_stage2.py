"""
Stage 2: train E_p (prosody residual) — F0/N + duration losses only (no decode).

Loads stage1 ckpt; freezes E_a. Targets: ST2 F0/N calibrated to baseline Kokoro heads,
and d_gt (with BOS/EOS) for duration.

  .venv\\Scripts\\python.exe -m timbre_encoder.kokoro_embed_train.train_stage2 \\
      --stage1-ckpt timbre_encoder/runs/kokoro_embed/stage1/stage1_final.pt \\
      --manifest timbre_encoder/manifests/dev-clean.filtered.csv --steps 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

from kokoro.pipeline import KPipeline  # noqa: E402
from kokoro_revers.model import KModel  # noqa: E402

from timbre_encoder.dual_encoder import (  # noqa: E402
    DualStyleEncoder,
    MIN_STYLE_FRAMES,
    crop_mel_for_style,
)
from gt_cache.utils import DEFAULT_FEATURES_ROOT  # noqa: E402
from timbre_encoder.kokoro_embed_train.common import (  # noqa: E402
    DEFAULT_BASELINE,
    MAX_WAV_SEC,
    calibrate_st2_to_kokoro,
    corr,
    d_gt_target_with_bos_eos,
    freeze_module,
    kokoro_f0n_for_ref,
    load_ckpt,
    pack_slot,
    prepare_utt,
    prosody_heads_forward_grad,
    save_ckpt,
    to_1d,
)
from timbre_encoder.librispeech_dataset import LibriTTSDataset  # noqa: E402


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device} baseline={args.baseline}", flush=True)

    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    kmodel = KModel(repo_id="hexgrad/Kokoro-82M").to(device)
    freeze_module(kmodel)
    # Weights stay frozen, but CuDNN LSTM needs train() for backward into ref_s.
    kmodel.predictor.train()
    kmodel.bert.eval()
    kmodel.bert_encoder.eval()
    kmodel.text_encoder.eval()
    kmodel.decoder.eval()

    baseline_pack = pipeline.load_voice(args.baseline).to(device)
    encoders = DualStyleEncoder().to(device)
    ckpt = load_ckpt(args.stage1_ckpt, map_location=device)
    encoders.load_state_dict(ckpt["encoders"])
    if "residual_scale" in ckpt:
        encoders.residual_scale = float(ckpt["residual_scale"])
    print(f"loaded stage1 {args.stage1_ckpt} step={ckpt.get('step')}", flush=True)

    for p in encoders.acoustic.parameters():
        p.requires_grad_(False)
    encoders.acoustic.eval()
    encoders.prosodic.train()
    opt = torch.optim.Adam(
        [p for p in encoders.prosodic.parameters() if p.requires_grad],
        lr=args.lr,
    )

    speakers = [s.strip() for s in args.speaker.split(",") if s.strip()] or None
    dataset = LibriTTSDataset(
        args.manifest,
        pipeline,
        max_wav_sec=args.max_wav_sec,
        require_alignment=True,
        load_f0n=True,
        require_f0n=True,
        features_root=args.features_root,
        speaker_ids=speakers,
    )
    print(f"rows={len(dataset)} speaker={speakers or 'ALL'}", flush=True)
    if len(dataset) == 0:
        raise SystemExit("No rows after speaker/align/f0n filters")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    step = 0

    while step < args.steps:
        indices = torch.randperm(len(dataset)).tolist()
        for idx in indices:
            if step >= args.steps:
                break
            item = dataset[idx]
            if item is None:
                continue
            try:
                phonemes = (item.get("phonemes_st2") or item["phonemes"]).strip()
                d_gt = item.get("d_gt")
                f0_st2 = item.get("f0")
                n_st2 = item.get("n")
                if d_gt is None or f0_st2 is None or n_st2 is None:
                    continue
                prep = prepare_utt(kmodel, phonemes, d_gt, device)
                if prep is None:
                    continue
                input_ids, aln, n_ph = prep
                mel = item["mel"].to(device)
                base_ref = pack_slot(baseline_pack, n_ph)

                mel_crop = crop_mel_for_style(
                    mel.unsqueeze(0), max_frames=args.style_frames, min_frames=args.min_style_frames
                )
                if mel_crop.shape[-1] < MIN_STYLE_FRAMES:
                    continue

                with torch.no_grad():
                    s_a, _ = encoders(mel_crop, base_ref, timbre_only=True)
                # train prosody residual
                da, dp = encoders.forward_deltas(mel_crop)
                s_a = s_a.detach()
                s_p = base_ref[:, 128:] + encoders.residual_scale * torch.tanh(dp)
                ref_s = torch.cat([s_a, s_p], dim=-1)

                pred_dur, f0_pred, n_pred = prosody_heads_forward_grad(
                    kmodel, input_ids, ref_s, aln
                )
                dur_tgt = d_gt_target_with_bos_eos(d_gt, device)
                if pred_dur.numel() != dur_tgt.numel():
                    continue
                loss_dur = F.l1_loss(pred_dur, dur_tgt)

                f0_ref, n_ref = kokoro_f0n_for_ref(kmodel, input_ids, base_ref, aln)
                f0_cal, n_cal = calibrate_st2_to_kokoro(
                    f0_st2.to(device), n_st2.to(device), f0_ref, n_ref
                )
                f0_p = to_1d(f0_pred)
                n_p = to_1d(n_pred)
                t = min(f0_p.numel(), f0_cal.numel())
                loss_f0 = F.l1_loss(f0_p[:t], f0_cal[:t].detach())
                tn = min(n_p.numel(), n_cal.numel())
                loss_n = F.l1_loss(n_p[:tn], n_cal[:tn].detach())

                loss = (
                    args.lambda_f0 * loss_f0
                    + args.lambda_n * loss_n
                    + args.lambda_dur * loss_dur
                )

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoders.prosodic.parameters(), args.grad_clip)
                opt.step()

                step += 1
                row = {
                    "step": step,
                    "loss": float(loss.detach()),
                    "f0": float(loss_f0.detach()),
                    "n": float(loss_n.detach()),
                    "dur": float(loss_dur.detach()),
                    "corr_f0": corr(f0_p[:t].detach(), f0_cal[:t], f0_cal[:t] > 50),
                    "corr_n": corr(n_p[:tn].detach(), n_cal[:tn]),
                }
                history.append(row)
                if step % args.log_every == 0 or step == 1:
                    print(
                        f"step={step} loss={row['loss']:.4f} f0={row['f0']:.4f} "
                        f"n={row['n']:.4f} dur={row['dur']:.4f} "
                        f"corr_f0={row['corr_f0']:.3f} corr_n={row['corr_n']:.3f}",
                        flush=True,
                    )
                if step % args.save_every == 0:
                    path = out_dir / f"stage2_step{step}.pt"
                    save_ckpt(
                        path,
                        {
                            "step": step,
                            "encoders": encoders.state_dict(),
                            "baseline": args.baseline,
                            "speaker": speakers,
                            "residual_scale": encoders.residual_scale,
                            "stage1_ckpt": args.stage1_ckpt,
                            "args": vars(args),
                        },
                    )
                    print(f"saved {path}", flush=True)
            except Exception as e:
                print(f"skip idx={idx}: {e}", flush=True)
                continue

    final = out_dir / "stage2_final.pt"
    save_ckpt(
        final,
        {
            "step": step,
            "encoders": encoders.state_dict(),
            "baseline": args.baseline,
            "speaker": speakers,
            "residual_scale": encoders.residual_scale,
            "stage1_ckpt": args.stage1_ckpt,
            "args": vars(args),
        },
    )
    (out_dir / "history_stage2.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"DONE stage2 steps={step} -> {final}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kokoro embed train stage2 (E_p / F0 N dur)")
    p.add_argument("--stage1-ckpt", type=str, required=True)
    p.add_argument(
        "--manifest",
        type=str,
        default=str(ROOT / "timbre_encoder" / "manifests" / "dev-clean.filtered.csv"),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "timbre_encoder" / "runs" / "kokoro_embed" / "stage2"),
    )
    p.add_argument("--features-root", type=str, default=str(DEFAULT_FEATURES_ROOT))
    p.add_argument("--baseline", type=str, default=DEFAULT_BASELINE)
    p.add_argument(
        "--speaker",
        type=str,
        default="",
        help="Comma-separated speaker_id(s); empty = all",
    )
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-wav-sec", type=float, default=MAX_WAV_SEC)
    p.add_argument("--style-frames", type=int, default=192)
    p.add_argument("--min-style-frames", type=int, default=MIN_STYLE_FRAMES)
    p.add_argument("--lambda-f0", type=float, default=1.0)
    p.add_argument("--lambda-n", type=float, default=1.0)
    p.add_argument("--lambda-dur", type=float, default=1.0)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--device", type=str, default="")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
