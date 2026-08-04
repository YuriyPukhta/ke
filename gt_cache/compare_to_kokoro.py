"""
Compare cached F0 (JDC vs RMVPE) + N vs Kokoro F0Ntrain, and d_gt vs duration pred.

Writes PNG plots + index.html for a few LibriTTS samples.

  .venv\\Scripts\\python.exe -m gt_cache.compare_to_kokoro \\
      --speaker 6209 --n 6 --baseline af_heart
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
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

from kokoro.pipeline import KPipeline  # noqa: E402
from kokoro_revers.model import KModel  # noqa: E402

from gt_cache import DEFAULT_FEATURES_ROOT, feature_path, load_curve  # noqa: E402
from timbre_encoder.kokoro_embed_train.common import (  # noqa: E402
    DEFAULT_BASELINE,
    calibrate_f0_voiced,
    calibrate_st2_to_kokoro,
    corr,
    d_gt_target_with_bos_eos,
    freeze_module,
    pack_slot,
    prepare_utt,
    prosody_heads_forward,
    to_1d,
)
from timbre_encoder.librispeech_dataset import parse_d_gt  # noqa: E402


def pick_rows(
    manifest: Path,
    speaker: str | None,
    n: int,
    features_root: Path,
    max_phonemes: int,
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    with manifest.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if speaker and row.get("speaker_id") != speaker:
                continue
            if row.get("align_status") != "ok":
                continue
            if row.get("f0_status") != "ok" or row.get("n_status") != "ok":
                continue
            if row.get("f0_rmvpe_status") != "ok":
                continue
            ph = (row.get("phonemes_st2") or "").strip()
            d_gt = parse_d_gt(row.get("d_gt"))
            if not ph or not d_gt or len(d_gt) != len(ph):
                continue
            if not (2 <= len(ph) <= max_phonemes):
                continue
            ap = row["audio_path"]
            if not feature_path(ap, "f0", features_root).is_file():
                continue
            if not feature_path(ap, "n", features_root).is_file():
                continue
            if not feature_path(ap, "f0_rmvpe", features_root).is_file():
                continue
            out.append(row)
            if len(out) >= n:
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot JDC/RMVPE F0 + N + d_gt vs Kokoro")
    ap.add_argument(
        "--manifest",
        type=str,
        default=str(ROOT / "timbre_encoder" / "manifests" / "train-clean-100.filtered.csv"),
    )
    ap.add_argument("--features-root", type=str, default=str(DEFAULT_FEATURES_ROOT))
    ap.add_argument("--speaker", type=str, default="6209")
    ap.add_argument("--n", type=int, default=6, help="number of samples")
    ap.add_argument("--baseline", type=str, default=DEFAULT_BASELINE)
    ap.add_argument("--max-phonemes", type=int, default=80)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "gt_cache" / "results" / "compare_to_kokoro"),
    )
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    features_root = Path(args.features_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = pick_rows(
        Path(args.manifest),
        args.speaker or None,
        args.n,
        features_root,
        args.max_phonemes,
    )
    if not rows:
        raise SystemExit("no usable rows (need f0 + f0_rmvpe + n + d_gt ok)")

    print(f"device={device} samples={len(rows)} baseline={args.baseline} out={out}", flush=True)
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    model = KModel(repo_id="hexgrad/Kokoro-82M").to(device)
    freeze_module(model)
    pack = pipeline.load_voice(args.baseline).to(device)

    lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>GT cache vs Kokoro</title>",
        "<style>body{font-family:sans-serif;max-width:1100px;margin:1.5rem auto}"
        "img{width:100%;max-width:1050px}.card{margin:1.5rem 0;padding:1rem;"
        "border:1px solid #ccc}code{background:#f4f4f4;padding:0 .25rem}</style>",
        "</head><body>",
        "<h1>Cached F0/N/d_gt vs Kokoro predictors</h1>",
        f"<p>baseline=<code>{html_lib.escape(args.baseline)}</code> · "
        "forced <code>d_gt</code> alignment · "
        "F0/N calibrated to Kokoro units (voiced mean/std)</p>",
        "<p><b>JDC</b>=old ST2 pitch · <b>RMVPE</b>=new · "
        "<b>N</b>=log_norm · <b>Kokoro</b>=F0Ntrain / duration_proj</p>",
    ]

    summary_rows: list[str] = [
        "<table border='1' cellpadding='4' cellspacing='0'>",
        "<tr><th>#</th><th>utt</th><th>corr F0 JDC</th><th>corr F0 RMVPE</th>"
        "<th>corr N</th><th>corr dur</th></tr>",
    ]

    for i, row in enumerate(rows):
        ph = (row.get("phonemes_st2") or "").strip()
        d_gt = parse_d_gt(row.get("d_gt"))
        assert ph and d_gt
        prep = prepare_utt(model, ph, d_gt, device)
        if prep is None:
            print(f"skip {i}: prepare_utt failed", flush=True)
            continue
        input_ids, aln, n_ph = prep
        slot = pack_slot(pack, n_ph)
        pred_dur, f0_k, n_k = prosody_heads_forward(model, input_ids, slot, aln)
        f0_k = to_1d(f0_k).cpu()
        n_k = to_1d(n_k).cpu()
        pred_dur = to_1d(pred_dur).cpu()
        d_tgt = d_gt_target_with_bos_eos(d_gt, device="cpu")

        ap = row["audio_path"]
        f0_jdc = load_curve(feature_path(ap, "f0", features_root))
        f0_rmv = load_curve(feature_path(ap, "f0_rmvpe", features_root))
        n_st2 = load_curve(feature_path(ap, "n", features_root))
        if f0_jdc is None or f0_rmv is None or n_st2 is None:
            print(f"skip {i}: missing curves", flush=True)
            continue

        t = int(f0_k.numel())
        f0_jdc_c, n_c = calibrate_st2_to_kokoro(f0_jdc, n_st2, f0_k, n_k)
        f0_rmv_c = calibrate_f0_voiced(f0_rmv, f0_k)

        f0_jdc_c = to_1d(f0_jdc_c).cpu()
        f0_rmv_c = to_1d(f0_rmv_c).cpu()
        n_c = to_1d(n_c).cpu()

        vok = f0_k > 50.0
        c_jdc = corr(f0_jdc_c, f0_k, vok)
        c_rmv = corr(f0_rmv_c, f0_k, vok)
        c_n = corr(n_c, n_k)
        # duration: compare inner phonemes (skip BOS/EOS) if lengths match
        if pred_dur.numel() == d_tgt.numel():
            c_dur = corr(pred_dur[1:-1], d_tgt[1:-1].float())
            d_plot = d_tgt.float().numpy()
            p_plot = pred_dur.numpy()
            x_ph = np.arange(len(d_plot))
        else:
            c_dur = float("nan")
            d_plot = np.asarray(d_gt, dtype=np.float32)
            p_plot = pred_dur[1:-1].numpy() if pred_dur.numel() >= 2 else pred_dur.numpy()
            m = min(len(d_plot), len(p_plot))
            d_plot, p_plot = d_plot[:m], p_plot[:m]
            x_ph = np.arange(m)
            c_dur = corr(torch.from_numpy(p_plot), torch.from_numpy(d_plot))

        stem = f"{i:02d}_{row['speaker_id']}"
        fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
        tx = np.arange(t)

        ax = axes[0]
        ax.plot(tx, f0_k.numpy(), label="Kokoro F0Ntrain", color="#1f77b4", lw=1.6)
        ax.plot(tx, f0_jdc_c.numpy(), label=f"JDC cal (corr_v={c_jdc:.3f})", color="#ff7f0e", alpha=0.85)
        ax.plot(tx, f0_rmv_c.numpy(), label=f"RMVPE cal (corr_v={c_rmv:.3f})", color="#2ca02c", alpha=0.85)
        ax.set_ylabel("F0 (Kokoro units)")
        ax.set_title(f"{stem} · F0")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(tx, n_k.numpy(), label="Kokoro N", color="#1f77b4", lw=1.6)
        ax.plot(tx, n_c.numpy(), label=f"cache N cal (corr={c_n:.3f})", color="#d62728", alpha=0.85)
        ax.set_ylabel("N")
        ax.set_title("N / energy")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.plot(x_ph, d_plot, label="d_gt (+BOS/EOS if aligned)", color="#1f77b4", lw=1.6, marker="o", ms=3)
        ax.plot(x_ph, p_plot, label=f"Kokoro dur pred (corr={c_dur:.3f})", color="#9467bd", alpha=0.85, marker="s", ms=3)
        ax.set_xlabel("token index")
        ax.set_ylabel("frames")
        ax.set_title("Duration: d_gt vs predictor")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        png = out / f"{stem}_curves.png"
        fig.savefig(png, dpi=120)
        plt.close(fig)

        text = (row.get("text") or "")[:180]
        lines.append(f"<div class='card'><h3>{html_lib.escape(stem)}</h3>")
        lines.append(f"<p>{html_lib.escape(text)}</p>")
        lines.append(
            f"<p>corr F0 JDC={c_jdc:.3f} · RMVPE={c_rmv:.3f} · N={c_n:.3f} · dur={c_dur:.3f} "
            f"· T={t} · phones={n_ph}</p>"
        )
        lines.append(f"<img src='{png.name}' alt='curves'/>")
        lines.append("</div>")

        summary_rows.append(
            f"<tr><td>{i}</td><td>{html_lib.escape(stem)}</td>"
            f"<td>{c_jdc:.3f}</td><td>{c_rmv:.3f}</td>"
            f"<td>{c_n:.3f}</td><td>{c_dur:.3f}</td></tr>"
        )
        print(
            f"[{i}] {stem} F0 jdc={c_jdc:.3f} rmvpe={c_rmv:.3f} N={c_n:.3f} dur={c_dur:.3f}",
            flush=True,
        )

    summary_rows.append("</table>")
    lines.insert(8, "<h2>Summary</h2>")
    lines.insert(9, "\n".join(summary_rows))
    lines.append("</body></html>")
    index = out / "index.html"
    index.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {index}", flush=True)


if __name__ == "__main__":
    main()
