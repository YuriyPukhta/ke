"""
Stage 1 (DEPRECATED for Libri clone experiments): DualStyleEncoder residual path.

Do NOT use for new runs. Proven path is free vector opt:
  python -m timbre_encoder.optimize_speaker_ref_s --speaker 4195 --steps 300

Legacy: train E_a (timbre residual) through frozen Kokoro + forced alignment + mel L1.
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
    freeze_module,
    kokoro_f0n_for_ref,
    pack_slot,
    prepare_utt,
    save_ckpt,
)
from timbre_encoder.librispeech_dataset import LibriTTSDataset  # noqa: E402
from timbre_encoder.mel_preprocess import waveform_to_mel  # noqa: E402


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device} baseline={args.baseline}", flush=True)

    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    kmodel = KModel(repo_id="hexgrad/Kokoro-82M").to(device)
    freeze_module(kmodel)  # eval + no grad; stage1 never backprops through predictor RNN

    baseline_pack = pipeline.load_voice(args.baseline).to(device)
    encoders = DualStyleEncoder().to(device)
    encoders.train()
    # Stage1: only acoustic head
    for p in encoders.prosodic.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(
        [p for p in encoders.acoustic.parameters() if p.requires_grad],
        lr=args.lr,
    )

    speakers = [s.strip() for s in args.speaker.split(",") if s.strip()] or None
    dataset = LibriTTSDataset(
        args.manifest,
        pipeline,
        max_wav_sec=args.max_wav_sec,
        require_alignment=True,
        load_f0n=True,
        require_f0n=not args.no_f0n_override,
        features_root=args.features_root,
        speaker_ids=speakers,
    )
    print(
        f"rows={len(dataset)} speaker={speakers or 'ALL'} max_wav_sec={args.max_wav_sec}",
        flush=True,
    )
    if len(dataset) == 0:
        raise SystemExit("No rows after speaker/align filters")

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
                if d_gt is None:
                    continue
                prep = prepare_utt(kmodel, phonemes, d_gt, device)
                if prep is None:
                    continue
                input_ids, aln, n_ph = prep
                mel = item["mel"].to(device)
                waveform = item["waveform"].to(device)

                base_ref = pack_slot(baseline_pack, n_ph)
                mel_crop = crop_mel_for_style(
                    mel.unsqueeze(0), max_frames=args.style_frames, min_frames=args.min_style_frames
                )
                # Too short even after pad policy: skip quietly (StyleEncoder 5x5 needs T>=80)
                if mel_crop.shape[-1] < MIN_STYLE_FRAMES:
                    continue
                s_a, _ = encoders(mel_crop, base_ref, timbre_only=True)
                # Prosody half frozen + detached so mel grads hit only E_a via decoder AdaIN.
                # F0/N always injected as constants — predictor LSTM stays out of the graph.
                ref_s = torch.cat([s_a, base_ref[:, 128:].detach()], dim=-1)

                with torch.no_grad():
                    if args.no_f0n_override:
                        f0_ov, n_ov = kokoro_f0n_for_ref(
                            kmodel, input_ids, base_ref, aln
                        )
                    else:
                        f0_st2 = item.get("f0")
                        n_st2 = item.get("n")
                        if f0_st2 is None or n_st2 is None:
                            continue
                        f0_ref, n_ref = kokoro_f0n_for_ref(
                            kmodel, input_ids, base_ref, aln
                        )
                        f0_ov, n_ov = calibrate_st2_to_kokoro(
                            f0_st2.to(device), n_st2.to(device), f0_ref, n_ref
                        )

                audio, _, _, _ = kmodel.forward_with_alignment(
                    input_ids, ref_s, aln, f0_override=f0_ov, n_override=n_ov
                )
                if audio.numel() < 100:
                    continue
                pred_mel = waveform_to_mel(
                    audio.unsqueeze(0) if audio.dim() == 1 else audio.view(1, -1),
                    device=device,
                )
                tgt_mel = waveform_to_mel(waveform.unsqueeze(0), device=device)
                t = min(pred_mel.shape[-1], tgt_mel.shape[-1])
                loss_mel = F.l1_loss(pred_mel[..., :t], tgt_mel[..., :t].detach())
                loss_res = (s_a - base_ref[:, :128]).pow(2).mean()
                loss = args.lambda_mel * loss_mel + args.lambda_res * loss_res

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoders.acoustic.parameters(), args.grad_clip)
                opt.step()

                step += 1
                row = {
                    "step": step,
                    "loss": float(loss.detach()),
                    "mel": float(loss_mel.detach()),
                    "res": float(loss_res.detach()),
                }
                history.append(row)
                if step % args.log_every == 0 or step == 1:
                    print(
                        f"step={step} loss={row['loss']:.4f} mel={row['mel']:.4f} "
                        f"res={row['res']:.4f} spk={item['speaker_id']}",
                        flush=True,
                    )
                if step % args.save_every == 0:
                    path = out_dir / f"stage1_step{step}.pt"
                    save_ckpt(
                        path,
                        {
                            "step": step,
                            "encoders": encoders.state_dict(),
                            "baseline": args.baseline,
                            "speaker": speakers,
                            "residual_scale": encoders.residual_scale,
                            "args": vars(args),
                        },
                    )
                    print(f"saved {path}", flush=True)
            except Exception as e:
                print(f"skip idx={idx}: {e}", flush=True)
                continue

    final = out_dir / "stage1_final.pt"
    save_ckpt(
        final,
        {
            "step": step,
            "encoders": encoders.state_dict(),
            "baseline": args.baseline,
            "speaker": speakers,
            "residual_scale": encoders.residual_scale,
            "args": vars(args),
        },
    )
    (out_dir / "history_stage1.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"DONE stage1 steps={step} -> {final}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kokoro embed train stage1 (E_a / mel)")
    p.add_argument(
        "--manifest",
        type=str,
        default=str(ROOT / "timbre_encoder" / "manifests" / "dev-clean.filtered.csv"),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "timbre_encoder" / "runs" / "kokoro_embed" / "stage1"),
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
    p.add_argument("--lambda-mel", type=float, default=1.0)
    p.add_argument("--lambda-res", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--no-f0n-override", action="store_true")
    p.add_argument("--device", type=str, default="")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
