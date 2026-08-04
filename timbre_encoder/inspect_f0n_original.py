"""
Inspect F0_pred / N_pred from original Kokoro voice pack (af_heart).

  .venv\\Scripts\\python.exe -m timbre_encoder.inspect_f0n_original
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

from kokoro.pipeline import KPipeline  # noqa: E402
from kokoro_revers.model import KModel  # noqa: E402

from timbre_encoder.librispeech_dataset import parse_d_gt  # noqa: E402
from timbre_encoder.optimize_speaker_ref_s import (  # noqa: E402
    build_alignment,
    extract_f0_n,
    kokoro_ids,
    load_speaker_rows,
    load_wav,
    pack_slot,
)


def resize(x: torch.Tensor, T: int) -> torch.Tensor:
    x = x.float().view(1, 1, -1)
    return F.interpolate(x, size=T, mode="linear", align_corners=False).view(-1)


def stats(x: torch.Tensor) -> dict[str, float]:
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
        "p10": float(x.quantile(0.1)),
        "p50": float(x.quantile(0.5)),
        "p90": float(x.quantile(0.9)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", type=str, default="6209")
    ap.add_argument("--baseline", type=str, default="af_heart")
    ap.add_argument(
        "--manifest",
        type=str,
        default=str(ROOT / "timbre_encoder/manifests/train-clean-100.filtered.csv"),
    )
    ap.add_argument("--n-clips", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-phonemes", type=int, default=100)
    ap.add_argument("--max-wav-sec", type=float, default=6.0)
    ap.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "timbre_encoder/runs/f0n_original_kokoro"),
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} baseline={args.baseline} speaker={args.speaker}", flush=True)

    pipeline = KPipeline(lang_code="a", model=False)
    pack = pipeline.load_voice(args.baseline).to(device)
    if pack.dim() == 2:
        pack = pack.unsqueeze(1)

    model = KModel(repo_id="hexgrad/Kokoro-82M").to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    rows = load_speaker_rows(
        Path(args.manifest),
        args.speaker,
        max_phonemes=args.max_phonemes,
        max_clips=args.n_clips + 4,
        seed=args.seed,
    )
    summary: list[dict] = []

    with torch.no_grad():
        for i, row in enumerate(rows[: args.n_clips]):
            phonemes = (row.get("phonemes_st2") or "").strip()
            d_gt = parse_d_gt(row.get("d_gt"))
            input_ids, ids, dropped = kokoro_ids(model, phonemes, device)
            if dropped or len(ids) != len(phonemes):
                print(f"#{i} skip dropped={dropped}", flush=True)
                continue
            aln = build_alignment(d_gt, device)
            if aln.shape[1] != input_ids.shape[1]:
                print(f"#{i} skip align", flush=True)
                continue

            wave = load_wav(row["audio_path"], args.max_wav_sec)
            ref_s = pack_slot(pack, len(phonemes)).to(device)
            audio, f0, n, _ = model.forward_with_alignment(input_ids, ref_s, aln)
            f0 = f0.squeeze().float().cpu()
            n = n.squeeze().float().cpu()
            if f0.dim() > 1:
                f0 = f0[0]
            if n.dim() > 1:
                n = n[0]

            f0_wav, e_wav = extract_f0_n(wave.numpy())
            f0w = resize(f0_wav, f0.numel())
            ew = resize(e_wav, n.numel())

            text = (row.get("text") or "")[:120]
            item = {
                "i": i,
                "text": text,
                "phoneme_len": len(phonemes),
                "T": int(f0.numel()),
                "F0_pred": stats(f0),
                "N_pred": stats(n),
                "F0_wav_hz": stats(f0w),
                "E_wav": stats(ew),
                "audio_rms": float(audio.float().pow(2).mean().sqrt()),
                "ref_rms": float(wave.float().pow(2).mean().sqrt()),
            }
            summary.append(item)
            fp, np_ = item["F0_pred"], item["N_pred"]
            print(
                f"#{i} T={item['T']} "
                f"F0 mean/std/min/max={fp['mean']:.3f}/{fp['std']:.3f}/{fp['min']:.3f}/{fp['max']:.3f} "
                f"N mean/std/min/max={np_['mean']:.3f}/{np_['std']:.3f}/{np_['min']:.3f}/{np_['max']:.3f}",
                flush=True,
            )

            np.savez(
                out / f"{i:02d}_curves.npz",
                F0_pred=f0.numpy(),
                N_pred=n.numpy(),
                F0_wav=f0w.numpy(),
                E_wav=ew.numpy(),
            )

            fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
            t = np.arange(f0.numel())
            axes[0].plot(t, f0.numpy(), label="Kokoro F0_pred", color="#1f77b4")
            ax0b = axes[0].twinx()
            ax0b.plot(t, f0w.numpy(), label="wav yin Hz", color="#ff7f0e", alpha=0.7)
            axes[0].set_ylabel("F0_pred (model units)")
            ax0b.set_ylabel("wav F0 (Hz)")
            axes[0].set_title(f"#{i} [{args.baseline}] {text[:60]}")
            h0, l0 = axes[0].get_legend_handles_labels()
            h0b, l0b = ax0b.get_legend_handles_labels()
            axes[0].legend(h0 + h0b, l0 + l0b, loc="upper right")

            axes[1].plot(t, n.numpy(), label="Kokoro N_pred", color="#2ca02c")
            ax1b = axes[1].twinx()
            ax1b.plot(t, ew.numpy(), label="wav log-energy", color="#d62728", alpha=0.7)
            axes[1].set_ylabel("N_pred (model units)")
            ax1b.set_ylabel("wav log-energy")
            axes[1].set_xlabel("frame")
            h1, l1 = axes[1].get_legend_handles_labels()
            h1b, l1b = ax1b.get_legend_handles_labels()
            axes[1].legend(h1 + h1b, l1 + l1b, loc="upper right")
            fig.tight_layout()
            fig.savefig(out / f"{i:02d}_f0n.png", dpi=120)
            plt.close(fig)

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
