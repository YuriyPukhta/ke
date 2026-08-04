"""
Fair compare: Kokoro-generated wav → extractors vs native F0Ntrain.

Voices synthesize text; F0/N come from the *same* audio as Kokoro pred.
Shows JDC F0, RMVPE F0, log_norm N (calibrated) vs Kokoro curves.

  .venv\\Scripts\\python.exe -m gt_cache.compare_kokoro_voices \\
      --voices af_heart,am_michael,af_bella,bm_george --n-texts 2
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

from kokoro.pipeline import KPipeline  # noqa: E402
from kokoro_revers.model import KModel  # noqa: E402

from gt_cache.rmvpe_backend import extract_f0_rmvpe, load_rmvpe  # noqa: E402
from gt_cache.swiftf0_backend import extract_f0_swift, load_swiftf0  # noqa: E402
from gt_cache.utils import extract_f0, extract_n, load_pitch_extractor, wav_to_mel  # noqa: E402
from timbre_encoder.mel_preprocess import SAMPLE_RATE  # noqa: E402
from timbre_encoder.optimize_speaker_ref_s import pack_slot  # noqa: E402
from timbre_encoder.st2_runtime import patch_torch_load  # noqa: E402

DEFAULT_TEXTS = [
    "Hello, this is a short test of pitch and energy.",
    "The quick brown fox jumps over the lazy dog near the river.",
    "I would like to know whether the extracted contours match Kokoro.",
]


def resize(x: torch.Tensor, t: int) -> torch.Tensor:
    x = x.float().view(1, 1, -1)
    return F.interpolate(x, size=t, mode="linear", align_corners=False).view(-1)


def match_mean_std(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    s = src.float().view(-1)
    r = ref.float().view(-1).detach()
    if s.numel() != r.numel():
        s = resize(s, r.numel())
    return (s - s.mean()) / s.std().clamp_min(1e-5) * r.std().clamp_min(1e-5) + r.mean()


def calibrate_f0_voiced(
    f0_src: torch.Tensor,
    f0_kokoro: torch.Tensor,
    src_voiced: float = 1.0,
    kokoro_voiced: float = 50.0,
) -> torch.Tensor:
    y = resize(f0_src.float(), f0_kokoro.numel())
    k = f0_kokoro.float().view(-1)
    y_v = y > src_voiced
    k_v = k > kokoro_voiced
    if int(y_v.sum()) >= 4 and int(k_v.sum()) >= 4:
        ys, ks = y[y_v], k[k_v]
        out = (y - ys.mean()) / ys.std().clamp_min(1e-5) * ks.std().clamp_min(1e-5) + ks.mean()
    else:
        out = match_mean_std(y, k)
    out = out.clone()
    out[~y_v] = 0.0
    return out


def corr(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    x, y = a.float().view(-1), b.float().view(-1)
    if mask is not None:
        m = mask.bool().view(-1)
        if int(m.sum()) < 4:
            return float("nan")
        x, y = x[m], y[m]
    if x.numel() != y.numel():
        n = min(x.numel(), y.numel())
        x, y = x[:n], y[:n]
    x, y = x - x.mean(), y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom) < 1e-8:
        return float("nan")
    return float((x * y).sum() / denom)


def to_1d(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().reshape(-1).cpu()


def pred_dur_to_alignment(pred_dur: torch.Tensor, device: torch.device) -> torch.Tensor:
    d = pred_dur.detach().float().view(-1).clamp_min(1)
    d_i = torch.round(d).long().clamp_min(1)
    n = int(d_i.numel())
    t = int(d_i.sum().item())
    aln = torch.zeros(1, n, t, device=device)
    col = 0
    for i in range(n):
        span = int(d_i[i].item())
        aln[0, i, col : col + span] = 1.0
        col += span
    return aln


def phonemes_from_text(pipeline: KPipeline, text: str) -> str | None:
    for _, ps, _ in pipeline.en_tokenize(pipeline.g2p(text)[1]):
        return ps
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Fair Kokoro-voice F0/N extractor compare")
    ap.add_argument(
        "--voices",
        type=str,
        default="af_heart,am_michael,af_bella,bm_george",
    )
    ap.add_argument("--n-texts", type=int, default=2)
    ap.add_argument("--texts", nargs="*", default=None)
    ap.add_argument("--thred", type=float, default=0.01, help="RMVPE UV threshold (lower=more sensitive)")
    ap.add_argument("--max-uv-gap", type=int, default=16)
    ap.add_argument("--median-k", type=int, default=7)
    ap.add_argument("--edge-fade", type=int, default=4)
    ap.add_argument("--swift-conf", type=float, default=0.55, help="SwiftF0 confidence_threshold")
    ap.add_argument("--swift-fmin", type=float, default=65.0)
    ap.add_argument("--swift-fmax", type=float, default=500.0)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "gt_cache" / "results" / "compare_kokoro_voices_swift"),
    )
    args = ap.parse_args()

    texts = args.texts if args.texts else DEFAULT_TEXTS[: max(1, args.n_texts)]
    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(
        f"device={device} voices={voices} texts={len(texts)} "
        f"thred={args.thred} uv_gap={args.max_uv_gap} med={args.median_k} "
        f"fade={args.edge_fade} out={out}",
        flush=True,
    )

    patch_torch_load()
    pitch_extractor = load_pitch_extractor(device)
    rmvpe, _ = load_rmvpe(device)
    swift = load_swiftf0(
        fmin=args.swift_fmin,
        fmax=args.swift_fmax,
        confidence_threshold=args.swift_conf,
    )

    pipeline = KPipeline(lang_code="a", model=False)
    model = KModel(repo_id="hexgrad/Kokoro-82M").to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    summary: list[dict] = []
    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Kokoro voices · F0/N fair compare</title>",
        "<style>body{font-family:sans-serif;max-width:1100px;margin:1.5rem auto}"
        "audio{width:100%}.card{margin:1.2rem 0;padding:1rem;border:1px solid #ccc}"
        "table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:.3rem .5rem}"
        "img{width:100%;max-width:1050px}</style></head><body>",
        "<h1>Fair: Kokoro wav → extractors vs F0Ntrain</h1>",
        "<p>Same generated audio. F0: <b>JDC</b>, <b>RMVPE sensitive</b>, <b>SwiftF0</b> "
        "(±smooth). N: <b>log_norm</b>. All calibrated to Kokoro units.</p>",
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
                    print(f"{voice}#{i} skip phonemes", flush=True)
                    continue
                input_ids = [
                    j for j in (model.vocab.get(p) for p in phonemes) if j is not None
                ]
                if len(input_ids) != len(phonemes) or len(input_ids) + 2 > model.context_length:
                    print(f"{voice}#{i} skip tokens", flush=True)
                    continue

                ids = torch.LongTensor([[0, *input_ids, 0]]).to(device)
                ref_s = pack_slot(pack, len(phonemes)).to(device)
                audio_nat, pred_dur = model.forward_with_tokens(ids, ref_s, speed=1.0)
                aln = pred_dur_to_alignment(pred_dur, device)
                _, f0_k, n_k, _ = model.forward_with_alignment(ids, ref_s, aln)
                f0_k = to_1d(f0_k)
                n_k = to_1d(n_k)
                wav = audio_nat.detach().cpu().float().view(-1).numpy()

                mel = wav_to_mel(wav, device=device)
                f0_jdc = extract_f0(pitch_extractor, mel).cpu()
                n_raw = extract_n(mel).cpu()
                f0_rmv_sens = extract_f0_rmvpe(
                    rmvpe,
                    wav,
                    int(mel.shape[-1]),
                    thred=args.thred,
                    smooth=True,
                    max_uv_gap=args.max_uv_gap,
                    median_k=args.median_k,
                    edge_fade=args.edge_fade,
                ).cpu()
                f0_sw = extract_f0_swift(
                    swift, wav, int(mel.shape[-1]), sample_rate=SAMPLE_RATE, smooth=False
                ).cpu()
                f0_sw_s = extract_f0_swift(
                    swift,
                    wav,
                    int(mel.shape[-1]),
                    sample_rate=SAMPLE_RATE,
                    smooth=True,
                    max_uv_gap=args.max_uv_gap,
                    median_k=args.median_k,
                    edge_fade=args.edge_fade,
                ).cpu()

                f0_jdc_c = calibrate_f0_voiced(f0_jdc, f0_k)
                f0_rmv_s = calibrate_f0_voiced(f0_rmv_sens, f0_k)
                f0_sw_c = calibrate_f0_voiced(f0_sw, f0_k)
                f0_sw_sc = calibrate_f0_voiced(f0_sw_s, f0_k)
                n_c = match_mean_std(n_raw, n_k)

                kok_v = f0_k > 50.0
                c_jdc = corr(f0_jdc_c, f0_k, kok_v)
                c_rmv_s = corr(f0_rmv_s, f0_k, kok_v)
                c_sw = corr(f0_sw_c, f0_k, kok_v)
                c_sw_s = corr(f0_sw_sc, f0_k, kok_v)
                c_n = corr(n_c, n_k)

                stem = f"{voice}_{i:02d}"
                sf.write(str(out / f"{stem}_native.wav"), wav, SAMPLE_RATE)

                t = np.arange(f0_k.numel())
                fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), constrained_layout=True)
                axes[0].plot(t, f0_k.numpy(), label="Kokoro F0Ntrain", color="#1f77b4", lw=1.8)
                axes[0].plot(
                    t, f0_jdc_c.numpy(), label=f"JDC ({c_jdc:.3f})", color="#ff7f0e", alpha=0.75
                )
                axes[0].plot(
                    t,
                    f0_rmv_s.numpy(),
                    label=f"RMVPE sens ({c_rmv_s:.3f})",
                    color="#9467bd",
                    alpha=0.7,
                    lw=1.2,
                )
                axes[0].plot(
                    t, f0_sw_c.numpy(), label=f"SwiftF0 ({c_sw:.3f})", color="#2ca02c", alpha=0.8, lw=1.3
                )
                axes[0].plot(
                    t,
                    f0_sw_sc.numpy(),
                    label=f"SwiftF0 smooth ({c_sw_s:.3f})",
                    color="#d62728",
                    alpha=0.85,
                    lw=1.5,
                )
                axes[0].set_ylabel("F0")
                axes[0].set_title(f"{voice} · {text[:70]}")
                axes[0].legend(loc="upper right", fontsize=8)
                axes[0].grid(True, alpha=0.3)

                axes[1].plot(t, n_k.numpy(), label="Kokoro N", color="#1f77b4", lw=1.6)
                axes[1].plot(
                    t, n_c.numpy(), label=f"log_norm cal (corr={c_n:.3f})", color="#d62728", alpha=0.85
                )
                axes[1].set_ylabel("N")
                axes[1].set_xlabel("frame")
                axes[1].legend(loc="upper right", fontsize=8)
                axes[1].grid(True, alpha=0.3)

                png = out / f"{stem}_curves.png"
                fig.savefig(png, dpi=120)
                plt.close(fig)

                item = {
                    "voice": voice,
                    "i": i,
                    "text": text,
                    "T": int(f0_k.numel()),
                    "corr_f0_jdc_voiced": c_jdc,
                    "corr_f0_rmvpe_smooth": c_rmv_s,
                    "corr_f0_swift": c_sw,
                    "corr_f0_swift_smooth": c_sw_s,
                    "corr_n": c_n,
                }
                summary.append(item)
                print(
                    f"{stem} jdc={c_jdc:.3f} rmvpe_s={c_rmv_s:.3f} "
                    f"swift={c_sw:.3f} swift_s={c_sw_s:.3f} N={c_n:.3f}",
                    flush=True,
                )

                html.append("<div class='card'>")
                html.append(f"<h3>{html_lib.escape(stem)}</h3>")
                html.append(f"<p>{html_lib.escape(text)}</p>")
                html.append(
                    f"<p>corr F0 JDC={c_jdc:.3f} · RMVPE sens={c_rmv_s:.3f} · "
                    f"SwiftF0={c_sw:.3f} · Swift smooth={c_sw_s:.3f} · N={c_n:.3f}</p>"
                )
                html.append(f"<p>native</p><audio controls src='{stem}_native.wav'></audio>")
                html.append(f"<img src='{png.name}' alt='curves'/>")
                html.append("</div>")

    # summary table
    rows_html = [
        "<h2>Summary</h2><table><tr><th>voice</th><th>#</th>"
        "<th>JDC</th><th>RMVPE sens</th><th>SwiftF0</th><th>Swift smooth</th>"
        "<th>N</th></tr>"
    ]
    for it in summary:
        rows_html.append(
            f"<tr><td>{html_lib.escape(it['voice'])}</td><td>{it['i']}</td>"
            f"<td>{it['corr_f0_jdc_voiced']:.3f}</td>"
            f"<td>{it['corr_f0_rmvpe_smooth']:.3f}</td>"
            f"<td>{it['corr_f0_swift']:.3f}</td>"
            f"<td>{it['corr_f0_swift_smooth']:.3f}</td>"
            f"<td>{it['corr_n']:.3f}</td></tr>"
        )
    rows_html.append("</table>")
    html.insert(6, "\n".join(rows_html))
    html.append("</body></html>")

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    index = out / "index.html"
    index.write_text("\n".join(html), encoding="utf-8")
    print(f"wrote {index}", flush=True)


if __name__ == "__main__":
    main()
