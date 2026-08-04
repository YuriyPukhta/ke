"""
Multi-voice F0/N calibration report: ST2 cache → Kokoro units.

Compares single-voice cal (each CORE voice) vs multi-voice mean reference.

  .venv\\Scripts\\python.exe -m timbre_encoder.kokoro_embed_train.report_f0n_calibration \\
      --manifest timbre_encoder/manifests/dev-clean.filtered.csv --limit 80
"""
from __future__ import annotations

import argparse
import csv
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
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

from kokoro.pipeline import KPipeline  # noqa: E402
from kokoro_revers.model import KModel  # noqa: E402

from gt_cache import feature_path, load_curve  # noqa: E402
from gt_cache.utils import DEFAULT_FEATURES_ROOT  # noqa: E402
from timbre_encoder.kokoro_embed_train.common import (  # noqa: E402
    DEFAULT_BASELINE,
    calibrate_st2_to_kokoro,
    corr,
    freeze_module,
    load_voice_packs,
    mae,
    multi_voice_kokoro_mean_f0n,
    pack_slot,
    prepare_utt,
    to_1d,
)
from timbre_encoder.voice_bank import CORE_VOICES  # noqa: E402
from timbre_encoder.librispeech_dataset import parse_d_gt  # noqa: E402
from timbre_encoder.mel_preprocess import SAMPLE_RATE  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-voice ST2→Kokoro F0/N cal report")
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--features-root", type=str, default=str(DEFAULT_FEATURES_ROOT))
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--voices", type=str, default=",".join(CORE_VOICES))
    ap.add_argument("--baseline", type=str, default=DEFAULT_BASELINE)
    ap.add_argument(
        "--out",
        type=str,
        default=str(HERE / "results" / "f0n_multivoice_cal"),
    )
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--listen", type=int, default=5, help="How many utts to synth for HTML")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    features_root = Path(args.features_root)
    voice_names = [v.strip() for v in args.voices.split(",") if v.strip()]

    print(f"device={device} voices={voice_names}", flush=True)
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    model = KModel(repo_id="hexgrad/Kokoro-82M").to(device)
    freeze_module(model)
    packs = load_voice_packs(pipeline, voice_names, device)

    rows: list[dict[str, str]] = []
    with open(args.manifest, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("align_status") != "ok":
                continue
            if row.get("f0_status") != "ok" or row.get("n_status") != "ok":
                continue
            rows.append(row)
            if args.limit > 0 and len(rows) >= args.limit:
                break
    print(f"utts={len(rows)}", flush=True)

    # metrics[voice_or_mean] = lists of corr_f0, corr_n, mae_f0, mae_n
    keys = list(voice_names) + ["multi_mean"]
    metrics: dict[str, dict[str, list[float]]] = {
        k: {"corr_f0": [], "corr_n": [], "mae_f0": [], "mae_n": []} for k in keys
    }
    # Also: after calibrating TO multi_mean, how well does cal match each voice's raw F0Ntrain?
    cal_vs_voice: dict[str, dict[str, list[float]]] = {
        v: {"corr_f0": [], "corr_n": []} for v in voice_names
    }
    cal_vs_mean: dict[str, list[float]] = {"corr_f0": [], "corr_n": []}

    listen_cards: list[str] = []
    n_ok = 0
    for i, row in enumerate(rows):
        ph = (row.get("phonemes_st2") or "").strip()
        d_gt = parse_d_gt(row.get("d_gt"))
        if not ph or not d_gt:
            continue
        f0_st2 = load_curve(feature_path(row["audio_path"], "f0", features_root))
        n_st2 = load_curve(feature_path(row["audio_path"], "n", features_root))
        if f0_st2 is None or n_st2 is None:
            continue
        prep = prepare_utt(model, ph, d_gt, device)
        if prep is None:
            continue
        input_ids, aln, n_ph = prep
        f0_mean, n_mean, per = multi_voice_kokoro_mean_f0n(
            model, input_ids, aln, packs, n_ph
        )

        # Per-voice: calibrate ST2 → that voice, measure vs that voice
        for name in voice_names:
            f0_v, n_v = per[name]
            f0_cal_v, n_cal_v = calibrate_st2_to_kokoro(f0_st2, n_st2, f0_v, n_v)
            metrics[name]["corr_f0"].append(corr(f0_cal_v, f0_v, f0_v > 50))
            metrics[name]["corr_n"].append(corr(n_cal_v, n_v))
            metrics[name]["mae_f0"].append(mae(f0_cal_v, f0_v))
            metrics[name]["mae_n"].append(mae(n_cal_v, n_v))

        # Multi-mean reference
        f0_cal_m, n_cal_m = calibrate_st2_to_kokoro(f0_st2, n_st2, f0_mean, n_mean)
        metrics["multi_mean"]["corr_f0"].append(corr(f0_cal_m, f0_mean, f0_mean > 50))
        metrics["multi_mean"]["corr_n"].append(corr(n_cal_m, n_mean))
        metrics["multi_mean"]["mae_f0"].append(mae(f0_cal_m, f0_mean))
        metrics["multi_mean"]["mae_n"].append(mae(n_cal_m, n_mean))
        cal_vs_mean["corr_f0"].append(corr(f0_cal_m, f0_mean, f0_mean > 50))
        cal_vs_mean["corr_n"].append(corr(n_cal_m, n_mean))

        # Stability: multi-cal vs each voice native curve
        for name in voice_names:
            f0_v, n_v = per[name]
            cal_vs_voice[name]["corr_f0"].append(corr(f0_cal_m, f0_v, f0_v > 50))
            cal_vs_voice[name]["corr_n"].append(corr(n_cal_m, n_v))

        n_ok += 1
        if n_ok <= args.listen:
            # plots + optional override listen with baseline pack
            stem = f"{n_ok:02d}_{row['speaker_id']}"
            fig, axes = plt.subplots(2, 2, figsize=(10, 6))
            t = np.arange(f0_mean.numel())
            axes[0, 0].plot(to_1d(f0_st2).numpy()[: len(t)], label="ST2 raw", alpha=0.7)
            axes[0, 0].plot(f0_cal_m.cpu().numpy(), label="cal→multi", alpha=0.85)
            axes[0, 0].plot(f0_mean.cpu().numpy(), label="multi mean", alpha=0.85)
            axes[0, 0].set_title("F0")
            axes[0, 0].legend(fontsize=8)
            axes[0, 1].plot(to_1d(n_st2).numpy()[: len(t)], label="ST2 raw", alpha=0.7)
            axes[0, 1].plot(n_cal_m.cpu().numpy(), label="cal→multi", alpha=0.85)
            axes[0, 1].plot(n_mean.cpu().numpy(), label="multi mean", alpha=0.85)
            axes[0, 1].set_title("N")
            axes[0, 1].legend(fontsize=8)
            # single baseline cal
            f0_b, n_b = per[args.baseline] if args.baseline in per else per[voice_names[0]]
            f0_cal_b, n_cal_b = calibrate_st2_to_kokoro(f0_st2, n_st2, f0_b, n_b)
            axes[1, 0].plot(f0_cal_b.cpu().numpy(), label=f"cal→{args.baseline}", alpha=0.85)
            axes[1, 0].plot(f0_b.cpu().numpy(), label=args.baseline, alpha=0.85)
            axes[1, 0].set_title(f"F0 single ({args.baseline})")
            axes[1, 0].legend(fontsize=8)
            axes[1, 1].plot(n_cal_b.cpu().numpy(), label=f"cal→{args.baseline}", alpha=0.85)
            axes[1, 1].plot(n_b.cpu().numpy(), label=args.baseline, alpha=0.85)
            axes[1, 1].set_title(f"N single ({args.baseline})")
            axes[1, 1].legend(fontsize=8)
            fig.tight_layout()
            png = out / f"{stem}_curves.png"
            fig.savefig(png, dpi=120)
            plt.close(fig)

            # synth listens: baseline pack + overrides
            base_pack = packs[args.baseline] if args.baseline in packs else packs[voice_names[0]]
            ref = pack_slot(base_pack, n_ph).to(device)
            try:
                import librosa

                wav, _ = librosa.load(row["audio_path"], sr=SAMPLE_RATE, mono=True)
                wav, _ = librosa.effects.trim(wav, top_db=30)
                max_s = int(8.0 * SAMPLE_RATE)
                if wav.shape[0] > max_s:
                    wav = wav[:max_s]
                sf.write(str(out / f"{stem}_00_original.wav"), wav, SAMPLE_RATE)

                audio_b, _, _, _ = model.forward_with_alignment(input_ids, ref, aln)
                sf.write(
                    str(out / f"{stem}_01_baseline.wav"),
                    audio_b.detach().cpu().numpy().reshape(-1),
                    SAMPLE_RATE,
                )
                audio_raw, _, _, _ = model.forward_with_alignment(
                    input_ids,
                    ref,
                    aln,
                    f0_override=resize_to(f0_st2, f0_b.numel()).to(device),
                    n_override=resize_to(n_st2, n_b.numel()).to(device),
                )
                sf.write(
                    str(out / f"{stem}_02_st2_raw.wav"),
                    audio_raw.detach().cpu().numpy().reshape(-1),
                    SAMPLE_RATE,
                )
                audio_sc, _, _, _ = model.forward_with_alignment(
                    input_ids,
                    ref,
                    aln,
                    f0_override=f0_cal_b.to(device),
                    n_override=n_cal_b.to(device),
                )
                sf.write(
                    str(out / f"{stem}_03_cal_single.wav"),
                    audio_sc.detach().cpu().numpy().reshape(-1),
                    SAMPLE_RATE,
                )
                audio_mc, _, _, _ = model.forward_with_alignment(
                    input_ids,
                    ref,
                    aln,
                    f0_override=f0_cal_m.to(device),
                    n_override=n_cal_m.to(device),
                )
                sf.write(
                    str(out / f"{stem}_04_cal_multi.wav"),
                    audio_mc.detach().cpu().numpy().reshape(-1),
                    SAMPLE_RATE,
                )
            except Exception as e:
                print(f"listen skip {stem}: {e}", flush=True)

            listen_cards.append(
                f"<div class='card'><h3>{html_lib.escape(stem)}</h3>"
                f"<p>{html_lib.escape(row.get('text','')[:120])}</p>"
                f"<img src='{png.name}' style='max-width:100%'/>"
                f"<p>original</p><audio controls src='{stem}_00_original.wav'></audio>"
                f"<p>baseline synth</p><audio controls src='{stem}_01_baseline.wav'></audio>"
                f"<p>ST2 raw override</p><audio controls src='{stem}_02_st2_raw.wav'></audio>"
                f"<p>cal single ({args.baseline})</p><audio controls src='{stem}_03_cal_single.wav'></audio>"
                f"<p>cal multi-mean</p><audio controls src='{stem}_04_cal_multi.wav'></audio>"
                f"</div>"
            )

        if (i + 1) % 20 == 0:
            print(f"  processed {i+1}/{len(rows)} ok={n_ok}", flush=True)

    def agg(xs: list[float]) -> dict[str, float]:
        a = [x for x in xs if x == x]  # drop nan
        if not a:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        return {"mean": float(np.mean(a)), "std": float(np.std(a)), "n": len(a)}

    summary = {
        "n_utts": n_ok,
        "voices": voice_names,
        "per_ref_cal": {k: {m: agg(v[m]) for m in v} for k, v in metrics.items()},
        "multi_cal_vs_each_voice": {
            v: {m: agg(cal_vs_voice[v][m]) for m in cal_vs_voice[v]} for v in voice_names
        },
        "multi_cal_vs_mean": {m: agg(cal_vs_mean[m]) for m in cal_vs_mean},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # HTML table
    rows_html = [
        "<tr><th>ref</th><th>corr_f0</th><th>corr_n</th><th>mae_f0</th><th>mae_n</th></tr>"
    ]
    for k in keys:
        pf = summary["per_ref_cal"][k]
        rows_html.append(
            "<tr>"
            f"<td>{html_lib.escape(k)}</td>"
            f"<td>{pf['corr_f0']['mean']:.3f}±{pf['corr_f0']['std']:.3f}</td>"
            f"<td>{pf['corr_n']['mean']:.3f}±{pf['corr_n']['std']:.3f}</td>"
            f"<td>{pf['mae_f0']['mean']:.2f}</td>"
            f"<td>{pf['mae_n']['mean']:.2f}</td>"
            "</tr>"
        )
    stab_rows = [
        "<tr><th>voice</th><th>multi_cal corr_f0 vs voice</th><th>corr_n</th></tr>"
    ]
    for v in voice_names:
        s = summary["multi_cal_vs_each_voice"][v]
        stab_rows.append(
            f"<tr><td>{v}</td><td>{s['corr_f0']['mean']:.3f}</td>"
            f"<td>{s['corr_n']['mean']:.3f}</td></tr>"
        )

    html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<title>F0/N multi-voice calibration</title>
<style>body{{font-family:sans-serif;max-width:1000px;margin:1.5rem auto;line-height:1.4}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:.3rem .5rem}}
.card{{margin:1rem 0;padding:1rem;border:1px solid #ddd}}audio{{width:100%}}</style></head><body>
<h1>ST2 → Kokoro F0/N calibration</h1>
<p>utts={n_ok}. Calibrate ST2 cache to each Kokoro voice (and multi-voice mean), measure match.</p>
<h2>Calibrate-to-ref quality</h2>
<table>{''.join(rows_html)}</table>
<h2>Multi-cal stability across voices</h2>
<table>{''.join(stab_rows)}</table>
<h2>Listen samples</h2>
{''.join(listen_cards)}
</body></html>"""
    (out / "index.html").write_text(html, encoding="utf-8")
    print(f"done ok={n_ok} -> {out / 'index.html'}", flush=True)
    print(json.dumps(summary["per_ref_cal"]["multi_mean"], indent=2))
    if args.baseline in summary["per_ref_cal"]:
        print(f"single {args.baseline}:", json.dumps(summary["per_ref_cal"][args.baseline], indent=2))


def resize_to(x: torch.Tensor, t: int) -> torch.Tensor:
    from timbre_encoder.kokoro_embed_train.common import resize_1d

    return resize_1d(to_1d(x), t)


if __name__ == "__main__":
    main()
