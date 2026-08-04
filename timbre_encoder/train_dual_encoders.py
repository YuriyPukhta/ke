"""
Train DualStyleEncoder only (E_a, E_p) on LibriTTS_R with frozen Kokoro + TMA + WavLM.

Example:
  python -m timbre_encoder.train_dual_encoders --manifest timbre_encoder/manifests/dev-clean.csv --steps 100
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

from kokoro.pipeline import KPipeline
from kokoro_revers.model import KModel

from timbre_encoder.aligner import FrozenTextAligner
from timbre_encoder.dual_encoder import DualStyleEncoder, crop_mel_for_style
from timbre_encoder.librispeech_dataset import LibriTTSDataset, text_to_kokoro_phonemes
from timbre_encoder.mel_preprocess import SAMPLE_RATE, waveform_to_mel
from timbre_encoder.slm_loss import WavLMFeatureLoss


def extract_f0_energy(waveform: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[torch.Tensor, torch.Tensor]:
    """Simple F0 / energy targets on mel hop grid (hop=300)."""
    hop = 300
    f0 = librosa.yin(waveform, fmin=50, fmax=500, sr=sr, frame_length=2048, hop_length=hop)
    f0 = np.nan_to_num(f0, nan=0.0)
    # energy from STFT
    S = np.abs(librosa.stft(waveform, n_fft=2048, hop_length=hop, win_length=1200))
    energy = np.log(np.clip(S.mean(axis=0), 1e-5, None))
    n = min(len(f0), len(energy))
    return torch.from_numpy(f0[:n].astype(np.float32)), torch.from_numpy(energy[:n].astype(np.float32))


def phonemes_to_input_ids(model: KModel, phonemes: str, device: torch.device) -> torch.Tensor:
    ids = list(filter(lambda i: i is not None, map(lambda p: model.vocab.get(p), phonemes)))
    return torch.LongTensor([[0, *ids, 0]]).to(device)


def freeze_module(m: torch.nn.Module) -> None:
    m.eval()
    for p in m.parameters():
        p.requires_grad = False


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}")

    pipeline = KPipeline(lang_code="a")
    kmodel = KModel(repo_id="hexgrad/Kokoro-82M").to(device)
    freeze_module(kmodel)

    encoders = DualStyleEncoder().to(device)
    encoders.train()
    opt = torch.optim.Adam(encoders.parameters(), lr=args.lr)

    print("Loading frozen Text Aligner...")
    aligner = FrozenTextAligner(device=device)

    # Kokoro-space baseline (residual target). Female default; keep prosody from same pack.
    baseline_pack = pipeline.load_voice(args.baseline).to(device)
    baseline_ref = baseline_pack[100, 0].unsqueeze(0)  # [1, 256] mid-length slot

    slm = None
    if not args.no_slm:
        print("Loading frozen WavLM...")
        slm = WavLMFeatureLoss(device=device)

    dataset = LibriTTSDataset(args.manifest, pipeline, max_wav_sec=args.max_wav_sec)
    print(f"manifest rows={len(dataset)} | residual on {args.baseline}")

    os.makedirs(args.out_dir, exist_ok=True)
    step = 0
    epoch = 0
    while step < args.steps:
        epoch += 1
        indices = torch.randperm(len(dataset)).tolist()
        for idx in indices:
            if step >= args.steps:
                break
            item = dataset[idx]
            if item is None:
                continue
            try:
                mel = item["mel"].to(device)  # [80, T]
                waveform = item["waveform"].to(device)
                phonemes = item["phonemes"]

                # Style crop -> residual encoders in Kokoro space
                mel_crop = crop_mel_for_style(mel.unsqueeze(0), max_frames=args.style_frames)
                s_a, s_p = encoders(mel_crop, baseline_ref, timbre_only=True)
                ref_s = torch.cat([s_a, s_p], dim=-1)

                # GT alignment from frozen TMA
                aln, d_full = aligner.alignment_for_kokoro(mel, phonemes)
                # Token count must match Kokoro input_ids
                input_ids = phonemes_to_input_ids(kmodel, phonemes, device)
                if aln.shape[1] != input_ids.shape[1]:
                    # Remap durations onto Kokoro token count
                    n_tok = input_ids.shape[1]
                    total = int(d_full.sum().item())
                    base = max(1, total // n_tok)
                    rem = total - base * n_tok
                    d_list = [base + (1 if i < rem else 0) for i in range(n_tok)]
                    aln = torch.zeros(1, n_tok, sum(d_list), device=device)
                    c = 0
                    for i, dur in enumerate(d_list):
                        aln[0, i, c : c + dur] = 1.0
                        c += dur

                audio_pred, f0_pred, n_pred, _ = kmodel.forward_with_alignment(input_ids, ref_s, aln)
                if audio_pred.dim() == 0 or audio_pred.numel() < 100:
                    continue

                # Mel reconstruction on aligned length
                pred_mel = waveform_to_mel(audio_pred.unsqueeze(0) if audio_pred.dim() == 1 else audio_pred)
                tgt_mel = waveform_to_mel(waveform.unsqueeze(0))
                t = min(pred_mel.shape[-1], tgt_mel.shape[-1])
                loss_mel = F.l1_loss(pred_mel[..., :t], tgt_mel[..., :t].detach())

                # F0 / energy (align lengths)
                f0_tgt, e_tgt = extract_f0_energy(waveform.detach().cpu().numpy())
                f0_tgt = f0_tgt.to(device)
                e_tgt = e_tgt.to(device)
                # F0_pred from model may be at decoder frame rate — interpolate to target
                f0_p = f0_pred.squeeze()
                n_p = n_pred.squeeze()
                if f0_p.dim() > 1:
                    f0_p = f0_p[0]
                if n_p.dim() > 1:
                    n_p = n_p[0]
                f0_p_i = F.interpolate(f0_p.view(1, 1, -1), size=f0_tgt.numel(), mode="linear", align_corners=False).view(-1)
                n_p_i = F.interpolate(n_p.view(1, 1, -1), size=e_tgt.numel(), mode="linear", align_corners=False).view(-1)
                # Kokoro F0 is in model units; map Hz target into similar scale
                f0_tgt_n = f0_tgt / 500.0
                f0_p_n = f0_p_i / (f0_p_i.detach().abs().mean() + 1e-3) * (f0_tgt_n.mean() + 1e-3)
                loss_f0 = F.l1_loss(f0_p_n, f0_tgt_n)
                e_tgt_n = (e_tgt - e_tgt.mean()) / (e_tgt.std() + 1e-3)
                n_p_n = (n_p_i - n_p_i.mean()) / (n_p_i.std() + 1e-3)
                loss_n = F.l1_loss(n_p_n, e_tgt_n)

                loss_slm = ref_s.new_zeros(())
                if slm is not None:
                    loss_slm = slm(audio_pred.view(1, -1), waveform.view(1, -1))

                # Keep residual small early — stay near Kokoro manifold
                loss_res = (s_a - baseline_ref[:, :128]).pow(2).mean()

                loss = (
                    args.lambda_mel * loss_mel
                    + args.lambda_f0 * loss_f0
                    + args.lambda_n * loss_n
                    + args.lambda_slm * loss_slm
                    + args.lambda_res * loss_res
                )

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoders.parameters(), 5.0)
                opt.step()

                step += 1
                if step % args.log_every == 0 or step == 1:
                    print(
                        f"step {step:05d} | loss={loss.item():.4f} "
                        f"mel={loss_mel.item():.4f} f0={loss_f0.item():.4f} "
                        f"n={loss_n.item():.4f} slm={loss_slm.detach().item():.4f} "
                        f"res={loss_res.item():.4f} | {item['speaker_id']}",
                        flush=True,
                    )
                if step % args.save_every == 0:
                    ckpt = Path(args.out_dir) / f"dual_encoder_step{step:05d}.pt"
                    torch.save(
                        {
                            "step": step,
                            "encoders": encoders.state_dict(),
                            "config": vars(args),
                            "residual_scale": encoders.residual_scale,
                            "baseline": args.baseline,
                        },
                        ckpt,
                    )
                    print(f"saved {ckpt}")
            except Exception as e:
                print(f"[skip] idx={idx} err={e}")
                continue

    final = Path(args.out_dir) / "dual_encoder_final.pt"
    torch.save(
        {
            "step": step,
            "encoders": encoders.state_dict(),
            "config": vars(args),
            "residual_scale": encoders.residual_scale,
            "baseline": args.baseline,
        },
        final,
    )
    print(f"done -> {final}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default=str(ROOT / "timbre_encoder" / "runs" / "dual"))
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--max-wav-sec", type=float, default=6.0)
    ap.add_argument("--style-frames", type=int, default=192)
    ap.add_argument("--baseline", type=str, default="af_heart")
    ap.add_argument("--lambda-mel", type=float, default=1.0)
    ap.add_argument("--lambda-f0", type=float, default=0.05)
    ap.add_argument("--lambda-n", type=float, default=0.05)
    ap.add_argument("--lambda-slm", type=float, default=0.2)
    ap.add_argument("--lambda-res", type=float, default=0.05, help="Stay near baseline timbre")
    ap.add_argument("--no-slm", action="store_true")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=200)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
