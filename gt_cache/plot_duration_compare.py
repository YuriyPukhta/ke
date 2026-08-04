"""
Plot Kokoro duration_proj vs real timing.

1) Kokoro voices: synthesize short text → pred_dur vs TMA align of the same wav
2) LibriSpeech/LibriTTS: cached d_gt vs duration_proj (baseline voice)

  .venv\\Scripts\\python.exe -m gt_cache.plot_duration_compare
"""
from __future__ import annotations

import argparse
import csv
import html as html_lib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

from kokoro.pipeline import KPipeline  # noqa: E402
from kokoro_revers.model import KModel  # noqa: E402

from timbre_encoder.aligner import FrozenTextAligner  # noqa: E402
from timbre_encoder.kokoro_embed_train.common import (  # noqa: E402
    DEFAULT_BASELINE,
    corr,
    d_gt_target_with_bos_eos,
    freeze_module,
    pack_slot,
    prepare_utt,
    prosody_heads_forward,
    to_1d,
)
from timbre_encoder.librispeech_dataset import parse_d_gt  # noqa: E402
from timbre_encoder.mel_preprocess import SAMPLE_RATE, waveform_to_mel  # noqa: E402
from timbre_encoder.phoneme_norm import normalize_kokoro_phonemes_for_st2  # noqa: E402

SHORT_TEXTS = [
    "Hello, this is a short test.",
    "The storm had become shapeless.",
]


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def kokoro_ids(model: KModel, phonemes: str, device: torch.device):
    ids = [model.vocab[p] for p in phonemes if p in model.vocab]
    return torch.LongTensor([[0, *ids, 0]]).to(device), ids


def plot_bars(ax, labels, gt, pred, title: str, corr_v: float):
    x = np.arange(len(gt))
    w = 0.4
    ax.bar(x - w / 2, gt, width=w, label="real / d_gt", color="#1f77b4", alpha=0.85)
    ax.bar(x + w / 2, pred, width=w, label="Kokoro pred", color="#9467bd", alpha=0.85)
    ax.plot(x, gt, color="#1f77b4", lw=1.2, marker="o", ms=3)
    ax.plot(x, pred, color="#9467bd", lw=1.2, marker="s", ms=3)
    if labels and len(labels) == len(gt) and len(gt) <= 40:
        ax.set_xticks(x)
        ax.set_xticklabels(list(labels), fontsize=7)
    ax.set_ylabel("frames")
    ax.set_title(f"{title}  ·  corr={corr_v:.3f}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)


def pick_libri(manifest: Path, speaker: str, n: int, max_ph: int) -> list[dict]:
    out = []
    with manifest.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if speaker and row.get("speaker_id") != speaker:
                continue
            if row.get("align_status") != "ok":
                continue
            ph = (row.get("phonemes_st2") or "").strip()
            d_gt = parse_d_gt(row.get("d_gt"))
            if not ph or not d_gt or len(d_gt) != len(ph):
                continue
            if not (8 <= len(ph) <= max_ph):
                continue
            out.append(row)
            if len(out) >= n:
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        type=str,
        default=str(ROOT / "timbre_encoder/manifests/train-clean-100.filtered.csv"),
    )
    ap.add_argument("--speaker", type=str, default="6209")
    ap.add_argument("--n-libri", type=int, default=3)
    ap.add_argument("--voices", type=str, default="af_heart,am_michael,af_bella")
    ap.add_argument("--baseline", type=str, default=DEFAULT_BASELINE)
    ap.add_argument("--max-phonemes", type=int, default=60)
    ap.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "gt_cache/results/duration_compare"),
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    voices = [v.strip() for v in args.voices.split(",") if v.strip()]

    print(f"device={device}", flush=True)
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    model = KModel(repo_id="hexgrad/Kokoro-82M").to(device).eval()
    freeze_module(model)
    aligner = FrozenTextAligner(device=device)

    cards: list[str] = []
    summary: list[str] = [
        "<table border='1' cellpadding='4' cellspacing='0'>",
        "<tr><th>kind</th><th>id</th><th>corr</th><th>MAE</th><th>sum_gt</th><th>sum_pred</th></tr>",
    ]

    # --- Kokoro voices: pred vs align(synth) ---
    for ti, text in enumerate(SHORT_TEXTS):
        graphemes, phonemes_raw, _ = next(pipeline(text, voice=voices[0], speed=1.0))
        ph = normalize_kokoro_phonemes_for_st2(phonemes_raw).normalized
        print(f"\n[voice text {ti}] {text!r}  ph_len={len(ph)}", flush=True)

        for voice in voices:
            pack = pipeline.load_voice(voice).to(device)
            input_ids, ids = kokoro_ids(model, ph, device)
            if len(ids) != len(ph):
                print(f"  skip {voice}: vocab drop", flush=True)
                continue
            pack_idx = min(max(len(ph) - 1, 0), pack.shape[0] - 1)
            ref_s = pack[pack_idx]
            if ref_s.dim() == 1:
                ref_s = ref_s.unsqueeze(0)

            with torch.no_grad():
                audio, pred_dur = model.forward_with_tokens(input_ids, ref_s, speed=1.0)
            pred = to_1d(pred_dur).cpu()
            pred_core = pred[1:-1]

            wav = audio.detach().float().cpu().numpy()
            wav_path = out / f"voice_{ti:02d}_{voice}.wav"
            sf.write(str(wav_path), wav, SAMPLE_RATE)

            mel = waveform_to_mel(torch.from_numpy(wav.astype(np.float32)), device=device).squeeze(0)
            _, d_gt_t, _ = aligner.align(mel, ph)
            d_gt = np.array([max(1, int(round(float(x)))) for x in d_gt_t.tolist()], dtype=np.float32)
            p = pred_core.numpy().astype(np.float32)
            m = min(len(d_gt), len(p))
            d_gt, p = d_gt[:m], p[:m]
            c = pearson(d_gt, p)
            mae = float(np.mean(np.abs(d_gt - p)))

            stem = f"voice_{ti:02d}_{voice}"
            fig, ax = plt.subplots(figsize=(11, 3.2), constrained_layout=True)
            plot_bars(ax, ph[:m], d_gt, p, f"{voice} · align(synth) vs pred", c)
            png = out / f"{stem}_dur.png"
            fig.savefig(png, dpi=130)
            plt.close(fig)

            cards.append(
                f"<div class='card'><h3>{html_lib.escape(stem)}</h3>"
                f"<p><b>Kokoro voice</b> · {html_lib.escape(text)}</p>"
                f"<p>corr={c:.3f} · MAE={mae:.2f} · sum_align={int(d_gt.sum())} · "
                f"sum_pred={int(p.sum())}</p>"
                f"<audio controls src='{wav_path.name}'></audio>"
                f"<img src='{png.name}'/></div>"
            )
            summary.append(
                f"<tr><td>kokoro</td><td>{html_lib.escape(stem)}</td>"
                f"<td>{c:.3f}</td><td>{mae:.2f}</td>"
                f"<td>{int(d_gt.sum())}</td><td>{int(p.sum())}</td></tr>"
            )
            print(f"  {stem}: corr={c:.3f} mae={mae:.2f}", flush=True)

    # --- LibriSpeech: d_gt vs pred ---
    rows = pick_libri(Path(args.manifest), args.speaker, args.n_libri, args.max_phonemes)
    print(f"\nLibri speaker={args.speaker} n={len(rows)}", flush=True)
    pack = pipeline.load_voice(args.baseline).to(device)

    for i, row in enumerate(rows):
        ph = row["phonemes_st2"].strip()
        d_gt = parse_d_gt(row["d_gt"])
        assert d_gt is not None
        prep = prepare_utt(model, ph, d_gt, device)
        if prep is None:
            continue
        input_ids, aln, n_ph = prep
        slot = pack_slot(pack, n_ph)
        pred_dur, _, _ = prosody_heads_forward(model, input_ids, slot, aln)
        pred = to_1d(pred_dur).cpu()
        d_tgt = d_gt_target_with_bos_eos(d_gt, device="cpu").float()

        if pred.numel() == d_tgt.numel():
            d_core = d_tgt[1:-1].numpy()
            p_core = pred[1:-1].numpy()
            c = float(corr(pred[1:-1], d_tgt[1:-1]))
        else:
            d_core = np.asarray(d_gt, dtype=np.float32)
            p_core = pred[1:-1].numpy() if pred.numel() >= 2 else pred.numpy()
            m = min(len(d_core), len(p_core))
            d_core, p_core = d_core[:m], p_core[:m]
            c = pearson(d_core, p_core)
        mae = float(np.mean(np.abs(d_core - p_core)))

        stem = f"libri_{i:02d}_{row['speaker_id']}"
        fig, ax = plt.subplots(figsize=(11, 3.2), constrained_layout=True)
        labels = ph if len(ph) == len(d_core) else None
        plot_bars(
            ax,
            labels,
            d_core,
            p_core,
            f"LibriTTS {row['speaker_id']} · d_gt vs {args.baseline} pred",
            c,
        )
        png = out / f"{stem}_dur.png"
        fig.savefig(png, dpi=130)
        plt.close(fig)

        text = (row.get("text") or "")[:160]
        cards.append(
            f"<div class='card'><h3>{html_lib.escape(stem)}</h3>"
            f"<p><b>LibriTTS</b> · {html_lib.escape(text)}</p>"
            f"<p>corr={c:.3f} · MAE={mae:.2f} · sum_d_gt={int(d_core.sum())} · "
            f"sum_pred={int(p_core.sum())} · phones={len(d_core)}</p>"
            f"<img src='{png.name}'/></div>"
        )
        summary.append(
            f"<tr><td>libri</td><td>{html_lib.escape(stem)}</td>"
            f"<td>{c:.3f}</td><td>{mae:.2f}</td>"
            f"<td>{int(d_core.sum())}</td><td>{int(p_core.sum())}</td></tr>"
        )
        print(f"  {stem}: corr={c:.3f} mae={mae:.2f} text={text!r}", flush=True)

    summary.append("</table>")
    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Duration: pred vs real</title>",
        "<style>body{font-family:sans-serif;max-width:1100px;margin:1.5rem auto}"
        "img{width:100%;max-width:1050px}.card{margin:1.5rem 0;padding:1rem;"
        "border:1px solid #ccc}code{background:#f4f4f4;padding:0 .25rem}</style>",
        "</head><body>",
        "<h1>Duration: Kokoro pred vs real timing</h1>",
        "<p><b>Kokoro voices</b>: TMA align of synthesized wav vs <code>duration_proj</code>.</p>",
        "<p><b>LibriTTS</b>: cached <code>d_gt</code> vs <code>duration_proj</code> "
        f"(baseline <code>{html_lib.escape(args.baseline)}</code>).</p>",
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
