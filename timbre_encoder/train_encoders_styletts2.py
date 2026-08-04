"""
Train acoustic/prosodic style encoders under frozen StyleTTS2 (clean path A).

  .venv\\Scripts\\python.exe -m timbre_encoder.train_encoders_styletts2 \\
      --manifest timbre_encoder/manifests/dev-clean.filtered.csv --steps 50

Only E_a / E_p train. Decoder, aligner, text encoder, pitch, PLBERT stay frozen.
Warm-start from StyleTTS2-LibriTTS style_encoder / predictor_encoder.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kokoro.pipeline import KPipeline

from timbre_encoder.dual_encoder import crop_mel_for_style
from timbre_encoder.encoder import StyleEncoder
from timbre_encoder.librispeech_dataset import LibriTTSDataset, text_to_kokoro_phonemes
from timbre_encoder.mel_preprocess import waveform_to_mel
from timbre_encoder.st2_runtime import (
    decode_with_style,
    load_frozen_st2,
    phonemes_to_st2_ids,
    prepare_asr_f0_n,
)


class AbsoluteDualStyleEncoder(nn.Module):
    """Two StyleEncoders → absolute s_a, s_p in StyleTTS2 space (no Kokoro residual)."""

    def __init__(self, dim_in: int = 64, style_dim: int = 128, max_conv_dim: int = 512):
        super().__init__()
        kw = dict(dim_in=dim_in, style_dim=style_dim, max_conv_dim=max_conv_dim)
        self.acoustic = StyleEncoder(**kw)
        self.prosodic = StyleEncoder(**kw)

    @classmethod
    def from_st2(cls, st2_model) -> "AbsoluteDualStyleEncoder":
        """Deep-copy ST2 style encoders (keeps their spectral_norm checkpoint layout)."""
        import copy

        m = cls.__new__(cls)
        nn.Module.__init__(m)
        m.acoustic = copy.deepcopy(st2_model.style_encoder)
        m.prosodic = copy.deepcopy(st2_model.predictor_encoder)
        for p in m.parameters():
            p.requires_grad = True
        return m

    def forward(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if mel.dim() == 3:
            x = mel.unsqueeze(1)
        elif mel.dim() == 4:
            x = mel
        else:
            raise ValueError(f"Expected mel [B,80,T] or [B,1,80,T], got {tuple(mel.shape)}")
        return self.acoustic(x), self.prosodic(x)


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}", flush=True)

    print("Loading frozen StyleTTS2-LibriTTS...", flush=True)
    st2, _ = load_frozen_st2(device)

    encoders = (
        AbsoluteDualStyleEncoder.from_st2(st2).to(device)
        if not args.no_warm_start
        else AbsoluteDualStyleEncoder().to(device)
    )
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        encoders.load_state_dict(ckpt["encoders"])
        start_step = int(ckpt.get("step", 0))
        print(f"resume {args.resume} step={start_step}", flush=True)
    elif not args.no_warm_start:
        print("warm-start: deepcopy ST2 style_encoder + predictor_encoder", flush=True)
    encoders.train()
    opt = torch.optim.Adam(encoders.parameters(), lr=args.lr)

    slm = None
    if not args.no_slm:
        from timbre_encoder.slm_loss import WavLMFeatureLoss

        print("Loading frozen WavLM...", flush=True)
        slm = WavLMFeatureLoss(device=device)

    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    dataset = LibriTTSDataset(args.manifest, pipeline, max_wav_sec=args.max_wav_sec)
    print(f"manifest rows={len(dataset)} max_wav_sec={args.max_wav_sec}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    history: list[dict] = []
    step = start_step
    target_steps = start_step + args.steps
    while step < target_steps:
        indices = torch.randperm(len(dataset)).tolist()
        for idx in indices:
            if step >= target_steps:
                break
            item = dataset[idx]
            if item is None:
                continue
            try:
                mel = item["mel"].to(device)  # [80, T] even-trimmed by waveform_to_mel
                waveform = item["waveform"].to(device)
                phonemes = item["phonemes"]
                text_ids = phonemes_to_st2_ids(phonemes, device)

                asr, F0, N, mel_even = prepare_asr_f0_n(st2, mel, text_ids)
                if asr.shape[-1] < 8:
                    continue

                mel_crop = crop_mel_for_style(
                    mel_even, max_frames=args.style_frames, min_frames=args.min_style_frames
                )
                if mel_crop.shape[-1] < 16 or mel_crop.shape[-2] < 16:
                    continue
                s_a, s_p = encoders(mel_crop)

                # Teacher styles from frozen ST2 (anchor — keeps AdaIN clean)
                with torch.no_grad():
                    s_a_t = st2.style_encoder(mel_crop.unsqueeze(1) if mel_crop.dim() == 3 else mel_crop)
                    s_p_t = st2.predictor_encoder(mel_crop.unsqueeze(1) if mel_crop.dim() == 3 else mel_crop)

                y = decode_with_style(st2, asr, F0, N, s_a)
                y = y.reshape(1, -1)  # [1, T] regardless of decoder squeeze quirks

                # Mel L1 vs GT (differentiable through decoder → s_a)
                mel_pred = waveform_to_mel(y.squeeze(0), device=device)
                mel_gt = mel_even
                t = min(mel_pred.shape[-1], mel_gt.shape[-1])
                loss_mel = F.l1_loss(mel_pred[..., :t], mel_gt[..., :t])

                loss_sa = F.l1_loss(s_a, s_a_t)
                loss_sp = F.l1_loss(s_p, s_p_t)
                # Cosine keeps direction on manifold (helps "dirty" AdaIN drift)
                loss_sa_cos = 1.0 - F.cosine_similarity(s_a, s_a_t, dim=-1).mean()

                loss = (
                    args.lambda_mel * loss_mel
                    + args.lambda_sa * loss_sa
                    + args.lambda_sa_cos * loss_sa_cos
                    + args.lambda_sp * loss_sp
                )

                if slm is not None:
                    t_wav = min(y.shape[-1], waveform.shape[-1])
                    loss_slm = slm(y[..., :t_wav], waveform[:t_wav].unsqueeze(0))
                    loss = loss + args.lambda_slm * loss_slm
                else:
                    loss_slm = None

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoders.parameters(), args.grad_clip)
                opt.step()

                step += 1
                row = {
                    "step": step,
                    "loss": float(loss.detach()),
                    "mel": float(loss_mel.detach()),
                    "sa": float(loss_sa.detach()),
                    "sa_cos": float(loss_sa_cos.detach()),
                    "sp": float(loss_sp.detach()),
                    "slm": float(loss_slm.detach()) if loss_slm is not None else None,
                    "s_a_norm": float(s_a.detach().norm()),
                }
                history.append(row)
                if step % args.log_every == 0 or step == start_step + 1:
                    slm_s = f" slm={row['slm']:.4f}" if row["slm"] is not None else ""
                    print(
                        f"step={step} loss={row['loss']:.4f} mel={row['mel']:.4f} "
                        f"sa={row['sa']:.4f} cos={row['sa_cos']:.4f} sp={row['sp']:.4f}"
                        f"{slm_s} s_a_norm={row['s_a_norm']:.3f}",
                        flush=True,
                    )
                if step % args.save_every == 0:
                    ckpt = {
                        "step": step,
                        "encoders": encoders.state_dict(),
                        "args": vars(args),
                    }
                    path = Path(args.out_dir) / f"encoders_step{step}.pt"
                    torch.save(ckpt, path)
                    print(f"saved {path}", flush=True)
            except Exception as e:
                print(f"skip idx={idx}: {e}", flush=True)
                continue

    final = Path(args.out_dir) / "encoders_final.pt"
    torch.save({"step": step, "encoders": encoders.state_dict(), "args": vars(args)}, final)
    (Path(args.out_dir) / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"DONE steps={step} -> {final}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train E_a/E_p under frozen StyleTTS2")
    p.add_argument(
        "--manifest",
        type=str,
        default=str(ROOT / "timbre_encoder" / "manifests" / "dev-clean.filtered.csv"),
    )
    p.add_argument("--out-dir", type=str, default=str(ROOT / "timbre_encoder" / "runs" / "st2_encoders"))
    p.add_argument("--resume", type=str, default="", help="Resume encoders_*.pt")
    p.add_argument("--steps", type=int, default=100, help="Additional steps from resume/start")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-wav-sec", type=float, default=4.0)
    p.add_argument("--style-frames", type=int, default=192)
    p.add_argument("--min-style-frames", type=int, default=64)
    p.add_argument("--lambda-mel", type=float, default=1.0)
    p.add_argument("--lambda-sa", type=float, default=0.1, help="L1 to frozen ST2 style_encoder")
    p.add_argument("--lambda-sa-cos", type=float, default=0.0, help="1-cos to frozen ST2 style_encoder")
    p.add_argument("--lambda-sp", type=float, default=1.0, help="L1 to frozen ST2 predictor_encoder")
    p.add_argument("--lambda-slm", type=float, default=0.2)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--no-warm-start", action="store_true")
    p.add_argument("--no-slm", action="store_true", help="Skip WavLM (recommended on 4GB GPU)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
