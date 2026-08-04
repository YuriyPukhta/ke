"""
Compare baseline JDC vs fine-tuned JDC against Kokoro F0Ntrain (live synth).

Usage:
  .venv\\Scripts\\python.exe -m gt_cache.jdc_finetune.eval_compare \\
      --weights gt_cache/jdc_finetune/checkpoints/jdc_kokoro_best.t7
"""
from __future__ import annotations

import argparse
import html as html_lib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

from kokoro.pipeline import KPipeline  # noqa: E402
from kokoro_revers.model import KModel  # noqa: E402

from gt_cache.jdc_finetune import DEFAULT_CKPT_DIR, DEFAULT_RESULTS_DIR  # noqa: E402
from gt_cache.jdc_finetune.common import (  # noqa: E402
    DEFAULT_TEXTS,
    corr_voiced,
    phonemes_from_text,
    pred_dur_to_alignment,
    resize_1d,
    tokens_from_phonemes,
)
from gt_cache.utils import extract_f0, load_pitch_extractor, wav_to_mel  # noqa: E402
from timbre_encoder.optimize_speaker_ref_s import pack_slot  # noqa: E402
from timbre_encoder.st2_runtime import patch_torch_load  # noqa: E402
from timbre_encoder.mel_preprocess import SAMPLE_RATE  # noqa: E402


def load_finetuned(path: Path, device: torch.device):
    patch_torch_load()
    from models import load_F0_models

    pe = load_F0_models(str(path))
    pe.eval().to(device)
    for p in pe.parameters():
        p.requires_grad_(False)
    return pe


def main() -> None:
    ap = argparse.ArgumentParser(description="Eval fine-tuned JDC vs baseline on Kokoro F0")
    ap.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_CKPT_DIR / "jdc_kokoro_best.t7"),
    )
    ap.add_argument(
        "--voices",
        type=str,
        default="af_nicole,af_jessica,af_nova,am_adam,am_echo,am_onyx,bf_alice,bf_lily,bm_fable,bm_lewis",
    )
    ap.add_argument("--n-texts", type=int, default=3)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--out", type=str, default=str(DEFAULT_RESULTS_DIR / "eval_compare"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    texts = DEFAULT_TEXTS[: max(1, args.n_texts)]
    weights = Path(args.weights)
    if not weights.is_file():
        raise SystemExit(f"missing weights: {weights}")

    patch_torch_load()
    jdc_base = load_pitch_extractor(device)
    jdc_ft = load_finetuned(weights, device)

    pipeline = KPipeline(lang_code="a", model=False)
    model = KModel(repo_id="hexgrad/Kokoro-82M").to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    summary = []
    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>JDC finetune vs Kokoro F0</title>",
        "<style>body{font-family:sans-serif;max-width:1100px;margin:1.5rem auto}"
        "audio{width:100%}.card{margin:1.2rem 0;padding:1rem;border:1px solid #ccc}"
        "table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:.3rem .5rem}"
        "img{width:100%;max-width:1050px}</style></head><body>",
        "<h1>JDC baseline vs fine-tuned → Kokoro F0Ntrain</h1>",
        f"<p>weights=<code>{html_lib.escape(str(weights))}</code></p>",
    ]

    with torch.no_grad():
        for voice in voices:
            pack = pipeline.load_voice(voice).to(device)
            if pack.dim() == 2:
                pack = pack.unsqueeze(1)
            html.append(f"<h2>{html_lib.escape(voice)}</h2>")
            for i, text in enumerate(texts):
                phonemes = phonemes_from_text(pipeline, text)
                if not phonemes:
                    continue
                input_ids = tokens_from_phonemes(model, phonemes)
                if not input_ids:
                    continue
                ids = torch.LongTensor([[0, *input_ids, 0]]).to(device)
                ref_s = pack_slot(pack, len(phonemes)).to(device)
                audio, pred_dur = model.forward_with_tokens(ids, ref_s, speed=1.0)
                aln = pred_dur_to_alignment(pred_dur, device)
                _, f0_k, _, _ = model.forward_with_alignment(ids, ref_s, aln)
                f0_k = f0_k.float().reshape(-1).cpu()
                wav = audio.detach().cpu().float().view(-1).numpy()
                mel = wav_to_mel(wav, device=device)
                t = int(mel.shape[-1])
                f0_k = resize_1d(f0_k, t)

                f0_b = extract_f0(jdc_base, mel).cpu()
                f0_f = extract_f0(jdc_ft, mel).cpu()
                if f0_b.numel() != t:
                    f0_b = resize_1d(f0_b, t)
                if f0_f.numel() != t:
                    f0_f = resize_1d(f0_f, t)

                c_b = corr_voiced(f0_b, f0_k)
                c_f = corr_voiced(f0_f, f0_k)
                stem = f"{voice}_{i:02d}"
                sf.write(str(out / f"{stem}.wav"), wav, SAMPLE_RATE)

                fig, ax = plt.subplots(1, 1, figsize=(11, 3.8), constrained_layout=True)
                tt = np.arange(t)
                ax.plot(tt, f0_k.numpy(), label="Kokoro F0Ntrain", color="#1f77b4", lw=1.8)
                ax.plot(tt, f0_b.numpy(), label=f"JDC base ({c_b:.3f})", color="#ff7f0e", alpha=0.75)
                ax.plot(tt, f0_f.numpy(), label=f"JDC ft ({c_f:.3f})", color="#2ca02c", alpha=0.85)
                ax.set_title(f"{voice} · {text[:70]}")
                ax.set_ylabel("F0 (Kokoro units)")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                png = out / f"{stem}_curves.png"
                fig.savefig(png, dpi=120)
                plt.close(fig)

                item = {
                    "voice": voice,
                    "i": i,
                    "text": text,
                    "corr_base": c_b,
                    "corr_ft": c_f,
                    "delta": c_f - c_b if (c_f == c_f and c_b == c_b) else float("nan"),
                }
                summary.append(item)
                print(
                    f"{stem} base={c_b:.3f} ft={c_f:.3f} d={item['delta']:.3f}",
                    flush=True,
                )
                html.append("<div class='card'>")
                html.append(f"<h3>{html_lib.escape(stem)}</h3>")
                html.append(f"<p>{html_lib.escape(text)}</p>")
                html.append(
                    f"<p>corr base={c_b:.3f} · ft={c_f:.3f} · d={item['delta']:.3f}</p>"
                )
                html.append(f"<audio controls src='{stem}.wav'></audio>")
                html.append(f"<img src='{png.name}' alt='curves'/>")
                html.append("</div>")

    mean_b = float(np.nanmean([s["corr_base"] for s in summary]))
    mean_f = float(np.nanmean([s["corr_ft"] for s in summary]))
    table = [
        "<h2>Summary</h2>",
        f"<p>mean corr base={mean_b:.3f} · ft={mean_f:.3f} · d={mean_f - mean_b:.3f}</p>",
        "<table><tr><th>voice</th><th>#</th><th>base</th><th>ft</th><th>d</th></tr>",
    ]
    for s in summary:
        table.append(
            f"<tr><td>{html_lib.escape(s['voice'])}</td><td>{s['i']}</td>"
            f"<td>{s['corr_base']:.3f}</td><td>{s['corr_ft']:.3f}</td>"
            f"<td>{s['delta']:.3f}</td></tr>"
        )
    table.append("</table>")
    html.insert(5, "\n".join(table))
    html.append("</body></html>")

    (out / "summary.json").write_text(
        json.dumps({"mean_base": mean_b, "mean_ft": mean_f, "items": summary}, indent=2),
        encoding="utf-8",
    )
    index = out / "index.html"
    index.write_text("\n".join(html), encoding="utf-8")
    print(f"mean base={mean_b:.3f} ft={mean_f:.3f} wrote {index}", flush=True)


if __name__ == "__main__":
    main()
