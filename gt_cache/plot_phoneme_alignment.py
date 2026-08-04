"""
Compare Kokoro-generated phoneme durations vs TMA aligner on the same synth.

Pipeline:
  1) synthesize text with a Kokoro voice → remember duration_proj (per phoneme)
  2) run TMA aligner on the generated wav
  3) plot Kokoro timing vs aligner timing

  .venv\\Scripts\\python.exe -m gt_cache.plot_phoneme_alignment
"""
from __future__ import annotations

import argparse
import html as html_lib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

from kokoro.pipeline import KPipeline  # noqa: E402
from kokoro_revers.model import KModel  # noqa: E402

from gt_cache.utils import wav_to_mel  # noqa: E402
from timbre_encoder.aligner import FrozenTextAligner  # noqa: E402
from timbre_encoder.kokoro_embed_train.common import freeze_module, pack_slot, to_1d  # noqa: E402
from timbre_encoder.mel_preprocess import SAMPLE_RATE  # noqa: E402
from timbre_encoder.phoneme_norm import normalize_kokoro_phonemes_for_st2  # noqa: E402

DEFAULT_TEXTS = [
    "Hello, this is a short test.",
    "The storm had become shapeless.",
    "He continued to advance.",
]

CMAP_KOKORO = ListedColormap(["#f4f4f4", "#9467bd"])
CMAP_ALIGN = ListedColormap(["#f4f4f4", "#1f77b4"])


def phone_labels(phonemes: str) -> list[str]:
    return ["␣" if ch == " " else (ch if ch.strip() else "·") for ch in phonemes]


def durs_to_hard(durs: list[int]) -> np.ndarray:
    d = [max(0, int(x)) for x in durs]
    t = max(1, sum(d))
    a = np.zeros((len(d), t), dtype=np.float32)
    cur = 0
    for i, dur in enumerate(d):
        if dur > 0:
            a[i, cur : cur + dur] = 1.0
            cur += dur
    return a


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def draw_timeline(ax, durs: list[int], labels: list[str], colors, y: float, height: float) -> int:
    cur = 0
    for i, dur in enumerate(durs):
        if dur <= 0:
            continue
        ax.barh(
            y,
            dur,
            left=cur,
            height=height,
            color=colors[i % len(colors)],
            edgecolor="white",
            linewidth=0.35,
        )
        if dur >= 2 or len(durs) <= 36:
            ax.text(
                cur + dur / 2,
                y,
                labels[i],
                ha="center",
                va="center",
                fontsize=7 if len(durs) > 40 else 8,
            )
        cur += int(dur)
    return cur


def plot_compare(
    phonemes: str,
    d_kokoro: list[int],
    d_align: list[int],
    title: str,
    out_png: Path,
) -> dict:
    """Kokoro-generated durations vs aligner on the same synth wav."""
    labels = phone_labels(phonemes)
    n = min(len(phonemes), len(d_kokoro), len(d_align))
    labels = labels[:n]
    d_kokoro = [max(1, int(x)) for x in d_kokoro[:n]]
    d_align = [max(1, int(x)) for x in d_align[:n]]

    a_k = durs_to_hard(d_kokoro)
    a_a = durs_to_hard(d_align)
    corr = pearson(np.asarray(d_kokoro, float), np.asarray(d_align, float))
    mae = float(np.mean(np.abs(np.asarray(d_kokoro, float) - np.asarray(d_align, float))))

    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    gs = fig.add_gridspec(4, 1, height_ratios=[1.2, 1.2, 0.85, 1.0])

    # 1) Kokoro timing used at generation
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(
        a_k,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=CMAP_KOKORO,
        vmin=0,
        vmax=1,
        extent=(-0.5, a_k.shape[1] - 0.5, -0.5, n - 0.5),
    )
    ax0.set_yticks(np.arange(n))
    ax0.set_yticklabels(labels, fontsize=8)
    ax0.set_ylabel("phoneme")
    ax0.set_xlabel("frame")
    ax0.set_title(
        f"{title}\n"
        f"1) Kokoro duration_proj (saved at synth) · T={a_k.shape[1]} · sum={sum(d_kokoro)}"
    )

    # 2) Aligner on the generated wav
    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(
        a_a,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=CMAP_ALIGN,
        vmin=0,
        vmax=1,
        extent=(-0.5, a_a.shape[1] - 0.5, -0.5, n - 0.5),
    )
    ax1.set_yticks(np.arange(n))
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_ylabel("phoneme")
    ax1.set_xlabel("frame")
    ax1.set_title(
        f"2) TMA aligner on generated wav · T={a_a.shape[1]} · sum={sum(d_align)} · "
        f"corr={corr:.3f} · MAE={mae:.2f}"
    )

    # 3) dual timeline
    ax2 = fig.add_subplot(gs[2])
    palette = plt.cm.tab20(np.linspace(0, 1, max(n, 1)))
    t_k = draw_timeline(ax2, d_kokoro, labels, palette, y=1.0, height=0.7)
    t_a = draw_timeline(ax2, d_align, labels, palette, y=0.0, height=0.7)
    ax2.set_yticks([0.0, 1.0])
    ax2.set_yticklabels(["2) aligner", "1) Kokoro"], fontsize=9)
    ax2.set_xlim(0, max(t_k, t_a, 1))
    ax2.set_ylim(-0.55, 1.55)
    ax2.set_xlabel("frame")
    ax2.set_title("phoneme timelines (same colors = same phoneme)")
    ax2.grid(True, axis="x", alpha=0.25)

    # 4) per-phoneme bars
    ax3 = fig.add_subplot(gs[3])
    x = np.arange(n)
    w = 0.4
    ax3.bar(x - w / 2, d_kokoro, width=w, label="1) Kokoro synth", color="#9467bd", alpha=0.9)
    ax3.bar(x + w / 2, d_align, width=w, label="2) aligner on synth", color="#1f77b4", alpha=0.9)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=7)
    ax3.set_ylabel("frames")
    ax3.set_title("per-phoneme duration")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, axis="y", alpha=0.3)

    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return {
        "corr": corr,
        "mae": mae,
        "sum_kokoro": int(sum(d_kokoro)),
        "sum_align": int(sum(d_align)),
        "n": n,
    }


def kokoro_ids(model: KModel, phonemes: str, device: torch.device):
    ids = [model.vocab[p] for p in phonemes if p in model.vocab]
    return torch.LongTensor([[0, *ids, 0]]).to(device), ids


def synth_and_remember_durs(
    model: KModel,
    phonemes: str,
    ref_s: torch.Tensor,
    device: torch.device,
) -> tuple[list[int], np.ndarray]:
    """Generate audio; return (pred_core durations, wav)."""
    input_ids, ids = kokoro_ids(model, phonemes, device)
    if len(ids) != len(phonemes):
        raise RuntimeError(f"vocab drop: ids={len(ids)} ph={len(phonemes)}")
    if ref_s.dim() == 1:
        ref_s = ref_s.unsqueeze(0)
    with torch.no_grad():
        audio, pred_dur = model.forward_with_tokens(input_ids, ref_s, speed=1.0)
    pred = to_1d(pred_dur).cpu()
    # drop BOS/EOS — core phoneme lengths used for the phrase body
    d_kokoro = [max(1, int(round(float(x)))) for x in pred[1:-1].tolist()]
    wav = audio.detach().float().cpu().numpy().astype(np.float32)
    return d_kokoro, wav


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Kokoro synth durations vs TMA aligner on the same wav"
    )
    ap.add_argument("--voices", type=str, default="af_heart,am_michael,af_bella")
    ap.add_argument(
        "--texts",
        type=str,
        default="|".join(DEFAULT_TEXTS),
        help="phrases separated by |",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "gt_cache/results/phoneme_alignment"),
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    texts = [t.strip() for t in args.texts.split("|") if t.strip()]
    print(f"device={device}", flush=True)

    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    model = KModel(repo_id="hexgrad/Kokoro-82M").to(device).eval()
    freeze_module(model)
    aligner = FrozenTextAligner(device=device)

    cards: list[str] = []
    summary = [
        "<table border='1' cellpadding='4' cellspacing='0'>",
        "<tr><th>id</th><th>corr</th><th>MAE</th>"
        "<th>sum Kokoro</th><th>sum aligner</th></tr>",
    ]

    for ti, text in enumerate(texts):
        _, phonemes_raw, _ = next(pipeline(text, voice=voices[0], speed=1.0))
        ph = normalize_kokoro_phonemes_for_st2(phonemes_raw).normalized
        print(f"\n[{ti}] phones={len(ph)} text={text!r}", flush=True)

        for voice in voices:
            pack = pipeline.load_voice(voice).to(device)
            ref_s = pack_slot(pack, len(ph))

            # 1) generate + remember Kokoro phoneme lengths
            d_kokoro, wav = synth_and_remember_durs(model, ph, ref_s, device)

            # 2) aligner on the generated phrase
            mel = wav_to_mel(wav, device)
            _, d_align_t, _ = aligner.align(mel, ph)
            d_align = [max(1, int(round(float(x)))) for x in d_align_t.tolist()]

            stem = f"{ti:02d}_{voice}"
            wav_path = out / f"{stem}.wav"
            sf.write(str(wav_path), wav, SAMPLE_RATE)
            png = out / f"{stem}_align.png"

            stats = plot_compare(
                ph,
                d_kokoro,
                d_align,
                f"{voice} · {text[:56]}",
                png,
            )
            cards.append(
                f"<div class='card'><h3>{html_lib.escape(stem)}</h3>"
                f"<p>voice=<code>{html_lib.escape(voice)}</code></p>"
                f"<p>{html_lib.escape(text)}</p>"
                f"<p>1) Kokoro <code>duration_proj</code> at synth → "
                f"2) TMA aligner on that wav</p>"
                f"<p>corr={stats['corr']:.3f} · MAE={stats['mae']:.2f} · "
                f"sum_kokoro={stats['sum_kokoro']} · "
                f"sum_align={stats['sum_align']}</p>"
                f"<audio controls src='{wav_path.name}'></audio>"
                f"<img src='{png.name}'/></div>"
            )
            summary.append(
                f"<tr><td>{html_lib.escape(stem)}</td>"
                f"<td>{stats['corr']:.3f}</td><td>{stats['mae']:.2f}</td>"
                f"<td>{stats['sum_kokoro']}</td>"
                f"<td>{stats['sum_align']}</td></tr>"
            )
            print(
                f"  {stem}: corr={stats['corr']:.3f} mae={stats['mae']:.2f} "
                f"kokoro={stats['sum_kokoro']} align={stats['sum_align']}",
                flush=True,
            )

    summary.append("</table>")
    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Kokoro synth vs aligner</title>",
        "<style>body{font-family:sans-serif;max-width:1100px;margin:1.5rem auto}"
        "img{width:100%;max-width:1050px}.card{margin:1.5rem 0;padding:1rem;"
        "border:1px solid #ccc}code{background:#f4f4f4;padding:0 .25rem}</style>",
        "</head><body>",
        "<h1>Kokoro synth timing vs TMA aligner</h1>",
        "<ol>",
        "<li>Generate phrase with Kokoro voice → save <code>duration_proj</code> "
        "per phoneme</li>",
        "<li>Run TMA aligner on the <b>generated</b> wav</li>",
        "<li>Compare on the graph (purple = Kokoro, blue = aligner)</li>",
        "</ol>",
        "<h2>Summary</h2>",
        *summary,
        *cards,
        "</body></html>",
    ]
    index = out / "index.html"
    index.write_text("\n".join(html), encoding="utf-8")
    print(f"\nwrote {index}", flush=True)


if __name__ == "__main__":
    main()
