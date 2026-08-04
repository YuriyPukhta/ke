"""
Plot fine-tuned JDC vs baseline on:
  1) held-out val voices (Kokoro F0Ntrain targets from dataset)
  2) a few LibriTTS/LibriSpeech wavs (real audio + optional Kokoro F0 via d_gt)

  .venv\\Scripts\\python.exe -m gt_cache.jdc_finetune.plot_results
"""
from __future__ import annotations

import argparse
import csv
import html as html_lib
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

from kokoro.pipeline import KPipeline  # noqa: E402
from kokoro_revers.model import KModel  # noqa: E402

from gt_cache.jdc_finetune import DEFAULT_CKPT_DIR, DEFAULT_DATA_DIR, DEFAULT_RESULTS_DIR  # noqa: E402
from gt_cache.jdc_finetune.common import corr_voiced, read_jsonl, resize_1d  # noqa: E402
from gt_cache.utils import (  # noqa: E402
    extract_f0,
    load_pitch_extractor,
    load_wav_for_cache,
    wav_to_mel,
)
from timbre_encoder.kokoro_embed_train.common import (  # noqa: E402
    pack_slot,
    prepare_utt,
    prosody_heads_forward,
    to_1d,
)
from timbre_encoder.librispeech_dataset import parse_d_gt  # noqa: E402
from timbre_encoder.st2_runtime import patch_torch_load  # noqa: E402


def load_finetuned(path: Path, device: torch.device):
    patch_torch_load()
    from models import load_F0_models

    pe = load_F0_models(str(path))
    pe.eval().to(device)
    for p in pe.parameters():
        p.requires_grad_(False)
    return pe


def match_mean_std(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    s = src.float().view(-1)
    r = ref.float().view(-1)
    n = min(s.numel(), r.numel())
    s, r = s[:n], r[:n]
    return (s - s.mean()) / s.std().clamp_min(1e-5) * r.std().clamp_min(1e-5) + r.mean()


def save_curve_fig(
    out_png: Path,
    title: str,
    curves: list[tuple[str, np.ndarray, str, float]],
    ylabel: str = "F0",
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(11, 3.6), constrained_layout=True)
    for label, y, color, lw in curves:
        ax.plot(np.arange(len(y)), y, label=label, color=color, lw=lw, alpha=0.9)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("frame")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_val_dataset(
    *,
    data_root: Path,
    jdc_base,
    jdc_ft,
    device: torch.device,
    out: Path,
    n_per_voice: int,
    seed: int,
) -> list[dict]:
    rows = read_jsonl(data_root / "val.jsonl")
    by_voice: dict[str, list[dict]] = {}
    for r in rows:
        by_voice.setdefault(r["voice"], []).append(r)
    rng = random.Random(seed)
    items = []
    html_parts = ["<h2>Held-out val voices (Kokoro F0Ntrain target)</h2>"]
    for voice in sorted(by_voice):
        picks = by_voice[voice][:]
        rng.shuffle(picks)
        picks = picks[:n_per_voice]
        for r in picks:
            obj = torch.load(data_root / r["path"], map_location="cpu", weights_only=False)
            mel = obj["mel"].float().to(device)
            f0_t = obj["f0"].float().view(-1)
            with torch.no_grad():
                f0_b = extract_f0(jdc_base, mel).cpu()
                f0_f = extract_f0(jdc_ft, mel).cpu()
            t = int(f0_t.numel())
            f0_b = resize_1d(f0_b, t)
            f0_f = resize_1d(f0_f, t)
            c_b = corr_voiced(f0_b, f0_t)
            c_f = corr_voiced(f0_f, f0_t)
            stem = f"val_{r['id']}"
            png = out / f"{stem}.png"
            text = (r.get("text") or "")[:90]
            save_curve_fig(
                png,
                f"{r['id']}  base={c_b:.3f}  ft={c_f:.3f}",
                [
                    ("Kokoro F0Ntrain", f0_t.numpy(), "#1f77b4", 1.8),
                    (f"JDC base ({c_b:.3f})", f0_b.numpy(), "#ff7f0e", 1.2),
                    (f"JDC ft ({c_f:.3f})", f0_f.numpy(), "#2ca02c", 1.5),
                ],
                ylabel="F0 (Kokoro units)",
            )
            item = {
                "split": "val",
                "id": r["id"],
                "voice": voice,
                "corr_base": c_b,
                "corr_ft": c_f,
                "png": png.name,
                "text": text,
            }
            items.append(item)
            html_parts.append("<div class='card'>")
            html_parts.append(f"<h3>{html_lib.escape(r['id'])}</h3>")
            html_parts.append(f"<p>{html_lib.escape(text)}</p>")
            html_parts.append(
                f"<p>corr base={c_b:.3f} · ft={c_f:.3f} · d={c_f - c_b:.3f}</p>"
            )
            html_parts.append(f"<img src='{png.name}' alt='curves'/>")
            html_parts.append("</div>")
            print(
                f"val {r['id']} base={c_b:.3f} ft={c_f:.3f}",
                flush=True,
            )
    (out / "_val_section.html").write_text("\n".join(html_parts), encoding="utf-8")
    return items


def pick_libri_rows(manifest: Path, n: int, speakers: list[str] | None) -> list[dict]:
    """Prefer round-robin across speakers so plots are diverse."""
    buckets: dict[str, list[dict]] = {}
    with manifest.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            sid = row.get("speaker_id", "")
            if speakers and sid not in speakers:
                continue
            if row.get("align_status") != "ok":
                continue
            if row.get("f0_status") != "ok":
                continue
            ph = (row.get("phonemes_st2") or "").strip()
            d_gt = parse_d_gt(row.get("d_gt"))
            if not ph or not d_gt or len(d_gt) != len(ph):
                continue
            if not (8 <= len(ph) <= 80):
                continue
            ap = row.get("audio_path", "")
            if not ap or not Path(ap).is_file():
                continue
            buckets.setdefault(sid, []).append(row)

    if not buckets:
        return []
    keys = list(buckets.keys())
    out: list[dict] = []
    i = 0
    while len(out) < n and keys:
        k = keys[i % len(keys)]
        if buckets[k]:
            out.append(buckets[k].pop(0))
        else:
            keys = [x for x in keys if buckets[x]]
            continue
        i += 1
    return out


def plot_libri(
    *,
    manifest: Path,
    jdc_base,
    jdc_ft,
    model: KModel,
    pack: torch.Tensor,
    device: torch.device,
    out: Path,
    n: int,
    speakers: list[str],
) -> list[dict]:
    rows = pick_libri_rows(manifest, n, speakers or None)
    if not rows:
        # fallback: any speakers
        rows = pick_libri_rows(manifest, n, None)
    html_parts = ["<h2>LibriTTS / LibriSpeech clips (real wav)</h2>"]
    items = []
    for i, row in enumerate(rows):
        wav = load_wav_for_cache(row["audio_path"])
        if wav is None:
            continue
        mel = wav_to_mel(wav, device=device)
        with torch.no_grad():
            f0_b = extract_f0(jdc_base, mel).cpu()
            f0_f = extract_f0(jdc_ft, mel).cpu()

        prepared = prepare_utt(model, row["phonemes_st2"], parse_d_gt(row["d_gt"]), device)
        f0_k = None
        if prepared is not None:
            input_ids, aln, n_ph = prepared
            ref_s = pack_slot(pack, n_ph).to(device)
            _, f0_k, _ = prosody_heads_forward(model, input_ids, ref_s, aln)
            f0_k = to_1d(f0_k).cpu()

        t = int(mel.shape[-1])
        f0_b = resize_1d(f0_b, t)
        f0_f = resize_1d(f0_f, t)
        curves = []
        c_b = c_f = float("nan")
        if f0_k is not None:
            f0_k = resize_1d(f0_k, t)
            # base JDC is old scale — show calibrated for readability; ft should be Kokoro-ish
            f0_b_cal = match_mean_std(f0_b, f0_k)
            f0_b_cal = f0_b_cal.clone()
            f0_b_cal[f0_b < 1.0] = 0.0
            c_b = corr_voiced(f0_b_cal, f0_k)
            c_f = corr_voiced(f0_f, f0_k)
            curves.append(("Kokoro F0Ntrain (baseline+d_gt)", f0_k.numpy(), "#1f77b4", 1.8))
            curves.append((f"JDC base cal ({c_b:.3f})", f0_b_cal.numpy(), "#ff7f0e", 1.2))
            curves.append((f"JDC ft ({c_f:.3f})", f0_f.numpy(), "#2ca02c", 1.5))
            ylabel = "F0 (Kokoro units)"
        else:
            curves.append(("JDC base", f0_b.numpy(), "#ff7f0e", 1.2))
            curves.append(("JDC ft", f0_f.numpy(), "#2ca02c", 1.5))
            ylabel = "F0"

        sid = row.get("speaker_id", "?")
        stem = f"libri_{sid}_{i:02d}"
        png = out / f"{stem}.png"
        text = (row.get("text") or "")[:100]
        save_curve_fig(png, f"{stem} · {text[:50]}", curves, ylabel=ylabel)
        item = {
            "split": "libri",
            "id": stem,
            "speaker": sid,
            "corr_base": c_b,
            "corr_ft": c_f,
            "png": png.name,
            "text": text,
            "audio_path": row["audio_path"],
        }
        items.append(item)
        html_parts.append("<div class='card'>")
        html_parts.append(f"<h3>{html_lib.escape(stem)}</h3>")
        html_parts.append(f"<p>{html_lib.escape(text)}</p>")
        if f0_k is not None:
            html_parts.append(
                f"<p>corr base_cal={c_b:.3f} · ft={c_f:.3f} · d={c_f - c_b:.3f}</p>"
            )
        html_parts.append(f"<img src='{png.name}' alt='curves'/>")
        html_parts.append("</div>")
        print(f"libri {stem} base={c_b:.3f} ft={c_f:.3f}", flush=True)
    (out / "_libri_section.html").write_text("\n".join(html_parts), encoding="utf-8")
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot JDC ft results on val + LibriSpeech")
    ap.add_argument("--weights", type=str, default=str(DEFAULT_CKPT_DIR / "jdc_kokoro_best.t7"))
    ap.add_argument("--data", type=str, default=str(DEFAULT_DATA_DIR))
    ap.add_argument(
        "--manifest",
        type=str,
        default=str(ROOT / "timbre_encoder" / "manifests" / "dev-clean.filtered.csv"),
    )
    ap.add_argument("--speakers", type=str, default="6313,8297,1462")
    ap.add_argument("--n-val-per-voice", type=int, default=1)
    ap.add_argument("--n-libri", type=int, default=6)
    ap.add_argument("--baseline-voice", type=str, default="af_heart")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--out", type=str, default=str(DEFAULT_RESULTS_DIR / "plots"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    weights = Path(args.weights)
    if not weights.is_file():
        raise SystemExit(f"missing weights: {weights}")

    patch_torch_load()
    jdc_base = load_pitch_extractor(device)
    jdc_ft = load_finetuned(weights, device)

    val_items = plot_val_dataset(
        data_root=Path(args.data),
        jdc_base=jdc_base,
        jdc_ft=jdc_ft,
        device=device,
        out=out,
        n_per_voice=args.n_val_per_voice,
        seed=args.seed,
    )

    pipeline = KPipeline(lang_code="a", model=False)
    model = KModel(repo_id="hexgrad/Kokoro-82M").to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    pack = pipeline.load_voice(args.baseline_voice).to(device)
    if pack.dim() == 2:
        pack = pack.unsqueeze(1)

    speakers = [s.strip() for s in args.speakers.split(",") if s.strip()]
    libri_items = plot_libri(
        manifest=Path(args.manifest),
        jdc_base=jdc_base,
        jdc_ft=jdc_ft,
        model=model,
        pack=pack,
        device=device,
        out=out,
        n=args.n_libri,
        speakers=speakers,
    )

    all_items = val_items + libri_items
    mean_val_b = float(np.nanmean([x["corr_base"] for x in val_items])) if val_items else float("nan")
    mean_val_f = float(np.nanmean([x["corr_ft"] for x in val_items])) if val_items else float("nan")
    mean_lb = float(np.nanmean([x["corr_base"] for x in libri_items])) if libri_items else float("nan")
    mean_lf = float(np.nanmean([x["corr_ft"] for x in libri_items])) if libri_items else float("nan")

    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>JDC finetune plots</title>",
        "<style>body{font-family:sans-serif;max-width:1100px;margin:1.5rem auto}"
        ".card{margin:1rem 0;padding:1rem;border:1px solid #ccc}"
        "img{width:100%;max-width:1050px}</style></head><body>",
        "<h1>JDC fine-tune: val hold-out + LibriSpeech</h1>",
        f"<p>weights=<code>{html_lib.escape(str(weights))}</code></p>",
        f"<p><b>Val</b> mean corr base={mean_val_b:.3f} · ft={mean_val_f:.3f} · "
        f"d={mean_val_f - mean_val_b:.3f} (n={len(val_items)})</p>",
        f"<p><b>Libri</b> mean corr base_cal={mean_lb:.3f} · ft={mean_lf:.3f} · "
        f"d={mean_lf - mean_lb:.3f} (n={len(libri_items)}; Kokoro via af_heart+d_gt)</p>",
    ]
    for section in ("_val_section.html", "_libri_section.html"):
        p = out / section
        if p.is_file():
            html.append(p.read_text(encoding="utf-8"))
    html.append("</body></html>")
    index = out / "index.html"
    index.write_text("\n".join(html), encoding="utf-8")
    (out / "summary.json").write_text(
        json.dumps(
            {
                "mean_val_base": mean_val_b,
                "mean_val_ft": mean_val_f,
                "mean_libri_base": mean_lb,
                "mean_libri_ft": mean_lf,
                "items": all_items,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"val mean base={mean_val_b:.3f} ft={mean_val_f:.3f} | "
        f"libri mean base={mean_lb:.3f} ft={mean_lf:.3f}",
        flush=True,
    )
    print(f"wrote {index}", flush=True)


if __name__ == "__main__":
    main()
