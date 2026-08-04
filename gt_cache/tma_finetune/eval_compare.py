"""
Compare baseline TMA vs fine-tuned TMA against Kokoro duration_proj (live synth).

  .venv\\Scripts\\python.exe -m gt_cache.tma_finetune.eval_compare \\
      --weights gt_cache/tma_finetune/checkpoints/tma_kokoro_best.pth
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

from gt_cache.jdc_finetune.common import DEFAULT_TEXTS, tokens_from_phonemes  # noqa: E402
from gt_cache.tma_finetune import DEFAULT_CKPT_DIR, DEFAULT_RESULTS_DIR  # noqa: E402
from gt_cache.tma_finetune.common import (  # noqa: E402
    duration_corr,
    forward_soft_attn,
    load_asr_trainable,
    phonemes_st2_from_text,
    rescale_durs_to_t,
    soft_to_durs,
    text_ids_from_phonemes,
)
from gt_cache.utils import wav_to_mel  # noqa: E402
from timbre_encoder.mel_preprocess import SAMPLE_RATE  # noqa: E402
from timbre_encoder.optimize_speaker_ref_s import pack_slot  # noqa: E402


def load_asr_eval(weights: Path | None, device: torch.device):
    model = load_asr_trainable(device, weights=weights)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def plot_durs(d_k, d_base, d_ft, labels, title, out_png: Path, c_base, c_ft):
    n = len(d_k)
    x = np.arange(n)
    w = 0.28
    fig, ax = plt.subplots(figsize=(12, 3.4), constrained_layout=True)
    ax.bar(x - w, d_k, width=w, label="Kokoro duration_proj", color="#9467bd")
    ax.bar(x, d_base, width=w, label=f"TMA baseline (corr={c_base:.3f})", color="#7f7f7f")
    ax.bar(x + w, d_ft, width=w, label=f"TMA fine-tuned (corr={c_ft:.3f})", color="#1f77b4")
    if n <= 40:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("frames (mel_down)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_CKPT_DIR / "tma_kokoro_best.pth"),
    )
    ap.add_argument(
        "--voices",
        type=str,
        default="af_nicole,am_adam,bf_emma,bm_lewis",
    )
    ap.add_argument("--n-texts", type=int, default=2)
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

    asr_base = load_asr_eval(None, device)
    asr_ft = load_asr_eval(weights, device)

    pipeline = KPipeline(lang_code="a", model=False)
    model = KModel(repo_id="hexgrad/Kokoro-82M").to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    summary = []
    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>TMA finetune vs Kokoro dur</title>",
        "<style>body{font-family:sans-serif;max-width:1100px;margin:1.5rem auto}"
        "img{width:100%;max-width:1050px}.card{margin:1.2rem 0;padding:1rem;"
        "border:1px solid #ccc}table{border-collapse:collapse}"
        "td,th{border:1px solid #ddd;padding:.3rem .5rem}</style></head><body>",
        "<h1>TMA baseline vs fine-tuned → Kokoro duration_proj</h1>",
        f"<p>weights=<code>{html_lib.escape(str(weights))}</code></p>",
        "<p>Pipeline: synth with Kokoro → save <code>duration_proj</code> → "
        "align mel(wav) with baseline / fine-tuned TMA.</p>",
    ]

    with torch.no_grad():
        for voice in voices:
            pack = pipeline.load_voice(voice).to(device)
            if pack.dim() == 2:
                pack = pack.unsqueeze(1)
            html.append(f"<h2>{html_lib.escape(voice)}</h2>")
            for i, text in enumerate(texts):
                phonemes = phonemes_st2_from_text(pipeline, text)
                if not phonemes:
                    continue
                input_ids = tokens_from_phonemes(model, phonemes)
                if not input_ids:
                    continue
                text_ids = text_ids_from_phonemes(phonemes)
                if len(text_ids) != len(phonemes):
                    continue

                ids = torch.LongTensor([[0, *input_ids, 0]]).to(device)
                ref_s = pack_slot(pack, len(phonemes)).to(device)
                audio, pred_dur = model.forward_with_tokens(ids, ref_s, speed=1.0)
                d_k_raw = [
                    max(1, int(round(float(x))))
                    for x in pred_dur.detach().float().view(-1)[1:-1].tolist()
                ]
                if len(d_k_raw) != len(phonemes):
                    continue

                wav = audio.detach().cpu().float().view(-1).numpy()
                mel = wav_to_mel(wav, device=device)
                soft_b = forward_soft_attn(asr_base, mel, torch.tensor(text_ids, device=device))
                soft_f = forward_soft_attn(asr_ft, mel, torch.tensor(text_ids, device=device))
                d_base = soft_to_durs(soft_b)
                d_ft = soft_to_durs(soft_f)
                t_down = soft_b.shape[-1]
                d_k = rescale_durs_to_t(d_k_raw, t_down)

                n = min(len(d_k), len(d_base), len(d_ft), len(phonemes))
                d_k, d_base, d_ft = d_k[:n], d_base[:n], d_ft[:n]
                c_b = duration_corr(d_k, d_base)
                c_f = duration_corr(d_k, d_ft)
                labels = ["␣" if c == " " else c for c in phonemes[:n]]

                stem = f"{voice}_{i:02d}"
                wav_path = out / f"{stem}.wav"
                sf.write(str(wav_path), wav, SAMPLE_RATE)
                png = out / f"{stem}_durs.png"
                plot_durs(
                    d_k,
                    d_base,
                    d_ft,
                    labels,
                    f"{voice} · {text[:60]}",
                    png,
                    c_b,
                    c_f,
                )
                summary.append(
                    {
                        "id": stem,
                        "voice": voice,
                        "corr_base": c_b,
                        "corr_ft": c_f,
                        "sum_k": sum(d_k),
                        "sum_base": sum(d_base),
                        "sum_ft": sum(d_ft),
                    }
                )
                html.append(
                    f"<div class='card'><h3>{html_lib.escape(stem)}</h3>"
                    f"<p>{html_lib.escape(text)}</p>"
                    f"<p>corr baseline={c_b:.3f} · fine-tuned={c_f:.3f} · "
                    f"sum K/base/ft={sum(d_k)}/{sum(d_base)}/{sum(d_ft)}</p>"
                    f"<audio controls src='{wav_path.name}'></audio>"
                    f"<img src='{png.name}'/></div>"
                )
                print(
                    f"{stem}: corr_base={c_b:.3f} corr_ft={c_f:.3f}",
                    flush=True,
                )

    mean_b = float(np.nanmean([r["corr_base"] for r in summary]))
    mean_f = float(np.nanmean([r["corr_ft"] for r in summary]))
    html.insert(
        6,
        f"<p><b>mean corr</b> baseline={mean_b:.3f} · fine-tuned={mean_f:.3f}</p>",
    )
    html.append("</body></html>")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "index.html").write_text("\n".join(html), encoding="utf-8")
    print(f"mean corr base={mean_b:.3f} ft={mean_f:.3f}", flush=True)
    print(f"wrote {out / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
