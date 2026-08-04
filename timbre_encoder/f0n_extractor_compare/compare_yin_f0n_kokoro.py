"""
Compare Kokoro-native F0/N vs librosa YIN + log-energy on Kokoro-generated audio.

Pipeline per clip:
  text → Kokoro native voice → wav + internal F0_pred / N_pred
  wav  → librosa YIN + log-energy (raw)
  raw  → mean/std match to Kokoro units (calibrated)
  resynth with f0/n override: calibrated vs raw

  .venv\\Scripts\\python.exe -m timbre_encoder.f0n_extractor_compare.compare_yin_f0n_kokoro
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

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

from kokoro.pipeline import KPipeline  # noqa: E402
from kokoro_revers.model import KModel  # noqa: E402

from timbre_encoder.mel_preprocess import SAMPLE_RATE  # noqa: E402
from timbre_encoder.optimize_speaker_ref_s import (  # noqa: E402
    extract_f0_n,
    pack_slot,
)

DEFAULT_TEXTS = [
    "Hello, this is a short test of pitch and energy.",
    "The quick brown fox jumps over the lazy dog near the river.",
    "I would like to know whether the extracted contours match Kokoro.",
]


def resize(x: torch.Tensor, T: int) -> torch.Tensor:
    x = x.float().view(1, 1, -1)
    return F.interpolate(x, size=T, mode="linear", align_corners=False).view(-1)


def match_mean_std(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Affine map src → ref mean/std (Pearson corr unchanged)."""
    s = src.float().view(-1)
    r = ref.float().view(-1).detach()
    if s.numel() != r.numel():
        s = resize(s, r.numel())
    s_std = s.std().clamp_min(1e-5)
    r_std = r.std().clamp_min(1e-5)
    return (s - s.mean()) / s_std * r_std + r.mean()


def calibrate_f0_yin(
    f0_yin: torch.Tensor,
    f0_kokoro: torch.Tensor,
    yin_voiced_hz: float = 1.0,
    kokoro_voiced: float = 50.0,
) -> torch.Tensor:
    """
    Map YIN Hz into Kokoro F0 units using voiced-only mean/std,
    then zero out frames where YIN is unvoiced (Kokoro keeps ~0 there).
    """
    y = resize(f0_yin.float(), f0_kokoro.numel())
    k = f0_kokoro.float().view(-1)
    y_v = y > yin_voiced_hz
    k_v = k > kokoro_voiced
    if int(y_v.sum()) >= 4 and int(k_v.sum()) >= 4:
        ys, ks = y[y_v], k[k_v]
        out = (y - ys.mean()) / ys.std().clamp_min(1e-5) * ks.std().clamp_min(1e-5) + ks.mean()
    else:
        out = match_mean_std(y, k)
    out = out.clone()
    out[~y_v] = 0.0
    return out


def calibrate_n_energy(e_yin: torch.Tensor, n_kokoro: torch.Tensor) -> torch.Tensor:
    return match_mean_std(e_yin, n_kokoro)


def stats(x: torch.Tensor | np.ndarray) -> dict[str, float]:
    if isinstance(x, torch.Tensor):
        t = x.float().detach().cpu().view(-1)
    else:
        t = torch.from_numpy(np.asarray(x, dtype=np.float32)).view(-1)
    return {
        "mean": float(t.mean()),
        "std": float(t.std()),
        "min": float(t.min()),
        "max": float(t.max()),
        "p10": float(t.quantile(0.1)),
        "p50": float(t.quantile(0.5)),
        "p90": float(t.quantile(0.9)),
    }


def corr(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    x = a.float().view(-1)
    y = b.float().view(-1)
    if mask is not None:
        m = mask.bool().view(-1)
        if int(m.sum()) < 4:
            return float("nan")
        x, y = x[m], y[m]
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom) < 1e-8:
        return float("nan")
    return float((x * y).sum() / denom)


def rms(x: torch.Tensor) -> float:
    return float(x.float().pow(2).mean().sqrt())


def pred_dur_to_alignment(pred_dur: torch.Tensor, device: torch.device) -> torch.Tensor:
    d = pred_dur.long().view(-1).clamp(min=1)
    n_tok = d.numel()
    t = int(d.sum().item())
    aln = torch.zeros(1, n_tok, t, device=device)
    idx = 0
    for i, di in enumerate(d.tolist()):
        aln[0, i, idx : idx + di] = 1.0
        idx += di
    return aln


def to_1d(x: torch.Tensor) -> torch.Tensor:
    x = x.float().squeeze()
    if x.dim() > 1:
        x = x[0]
    return x.view(-1)


def phonemes_from_text(pipeline: KPipeline, text: str) -> str | None:
    # g2p → phoneme tokens; take first chunk from en_tokenize
    for _, ps, _ in pipeline.en_tokenize(pipeline.g2p(text)[1]):
        return ps
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Kokoro F0/N vs librosa YIN calibration test")
    ap.add_argument("--voices", type=str, default="af_heart,am_michael")
    ap.add_argument("--texts", type=str, nargs="*", default=None)
    ap.add_argument(
        "--out",
        type=str,
        default=str(HERE / "results" / "yin_f0n_kokoro_compare"),
    )
    args = ap.parse_args()

    texts = args.texts if args.texts else DEFAULT_TEXTS
    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} voices={voices} n_texts={len(texts)} out={out}", flush=True)

    pipeline = KPipeline(lang_code="a", model=False)
    model = KModel(repo_id="hexgrad/Kokoro-82M").to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    summary: list[dict] = []
    html: list[str] = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Kokoro F0/N vs YIN</title>",
        "<style>body{font-family:sans-serif;max-width:1100px;margin:1.5rem auto;line-height:1.4}"
        "audio{width:100%;margin:.15rem 0}.card{margin:1.2rem 0;padding:1rem;border:1px solid #ccc}"
        "code{background:#f4f4f4;padding:0 .25rem}table{border-collapse:collapse;font-size:13px}"
        "td,th{border:1px solid #ddd;padding:.25rem .5rem;text-align:right}"
        "th:first-child,td:first-child{text-align:left}"
        "img{max-width:100%;border:1px solid #eee;margin:.5rem 0}</style></head><body>",
        "<h1>Kokoro native F0/N vs librosa YIN + log-energy</h1>",
        "<p>Audio is synthesized with official Kokoro voices. "
        "Internal curves come from <code>F0Ntrain</code>. "
        "Library curves from YIN (Hz) + STFT log-energy on the <b>same</b> generated wav.</p>"
        "<ul>"
        "<li><b>raw</b> — resized only (Hz / log-E as-is)</li>"
        "<li><b>naive cal</b> — global mean/std match (Pearson unchanged vs raw)</li>"
        "<li><b>voiced-cal</b> — match on voiced stats; YIN-unvoiced frames forced to 0</li>"
        "</ul>"
        "<p>Note: Kokoro F0 is ~Hz on voiced frames but ≈0 (sometimes slightly negative) on UV. "
        "YIN invents pitch on silence → full-curve correlation looks worse than voiced-only.</p>",
    ]

    with torch.no_grad():
        for voice in voices:
            pack = pipeline.load_voice(voice).to(device)
            if pack.dim() == 2:
                pack = pack.unsqueeze(1)

            html.append(f"<h2>Voice: {html_lib.escape(voice)}</h2>")

            for i, text in enumerate(texts):
                phonemes = phonemes_from_text(pipeline, text)
                if not phonemes:
                    print(f"{voice}#{i} skip: no phonemes", flush=True)
                    continue

                input_ids = list(
                    filter(lambda j: j is not None, map(lambda p: model.vocab.get(p), phonemes))
                )
                if len(input_ids) != len(phonemes):
                    print(f"{voice}#{i} skip: vocab drop", flush=True)
                    continue
                if len(input_ids) + 2 > model.context_length:
                    print(f"{voice}#{i} skip: too long", flush=True)
                    continue

                ids = torch.LongTensor([[0, *input_ids, 0]]).to(device)
                ref_s = pack_slot(pack, len(phonemes)).to(device)

                audio_nat, pred_dur = model.forward_with_tokens(ids, ref_s, speed=1.0)
                aln = pred_dur_to_alignment(pred_dur, device)
                # Same path → internal F0/N used by decoder
                _, f0_k, n_k, _ = model.forward_with_alignment(ids, ref_s, aln)
                f0_k = to_1d(f0_k).cpu()
                n_k = to_1d(n_k).cpu()

                wav = audio_nat.detach().cpu().float().view(-1)
                f0_yin, e_yin = extract_f0_n(wav.numpy(), sr=SAMPLE_RATE)
                f0_raw = resize(f0_yin, f0_k.numel()).cpu()
                n_raw = resize(e_yin, n_k.numel()).cpu()
                # Naive full-curve affine (corr identical to raw — kept for listen A/B)
                f0_cal_naive = match_mean_std(f0_yin, f0_k).cpu()
                # Voiced-aware: match on voiced stats, zero where YIN unvoiced
                f0_cal = calibrate_f0_yin(f0_yin, f0_k).cpu()
                n_cal = calibrate_n_energy(e_yin, n_k).cpu()

                yin_v = f0_raw > 1.0
                kok_v = f0_k > 50.0
                both_v = yin_v & kok_v
                item = {
                    "voice": voice,
                    "i": i,
                    "text": text,
                    "phoneme_len": len(phonemes),
                    "T_f0": int(f0_k.numel()),
                    "T_n": int(n_k.numel()),
                    "frac_yin_voiced": float(yin_v.float().mean()),
                    "frac_kokoro_voiced": float(kok_v.float().mean()),
                    "kokoro_F0": stats(f0_k),
                    "kokoro_F0_voiced": stats(f0_k[kok_v]) if int(kok_v.sum()) else None,
                    "kokoro_N": stats(n_k),
                    "yin_raw_Hz": stats(f0_raw),
                    "energy_raw": stats(n_raw),
                    "yin_cal": stats(f0_cal),
                    "energy_cal": stats(n_cal),
                    # Pearson invariant to global affine → raw == naive-cal on full curve
                    "corr_f0_all": corr(f0_raw, f0_k),
                    "corr_f0_kokoro_voiced": corr(f0_raw, f0_k, kok_v),
                    "corr_f0_both_voiced": corr(f0_raw, f0_k, both_v),
                    "corr_f0_cal_voiced_aware": corr(f0_cal, f0_k),
                    "corr_n_raw": corr(n_raw, n_k),
                    "corr_n_cal": corr(n_cal, n_k),
                    "rms_native": rms(wav),
                }

                # Resynth: reinject calibrated / raw into same alignment
                audio_cal, _, _, _ = model.forward_with_alignment(
                    ids,
                    ref_s,
                    aln,
                    f0_override=f0_cal.to(device),
                    n_override=n_cal.to(device),
                )
                audio_raw, _, _, _ = model.forward_with_alignment(
                    ids,
                    ref_s,
                    aln,
                    f0_override=f0_raw.to(device),
                    n_override=n_raw.to(device),
                )
                audio_naive, _, _, _ = model.forward_with_alignment(
                    ids,
                    ref_s,
                    aln,
                    f0_override=f0_cal_naive.to(device),
                    n_override=n_cal.to(device),
                )
                # F0-only / N-only calibrated (isolate axes)
                audio_f0only, _, _, _ = model.forward_with_alignment(
                    ids,
                    ref_s,
                    aln,
                    f0_override=f0_cal.to(device),
                )
                audio_nonly, _, _, _ = model.forward_with_alignment(
                    ids,
                    ref_s,
                    aln,
                    n_override=n_cal.to(device),
                )

                item["rms_cal"] = rms(audio_cal)
                item["rms_raw"] = rms(audio_raw)
                item["rms_naive"] = rms(audio_naive)
                item["rms_f0only"] = rms(audio_f0only)
                item["rms_nonly"] = rms(audio_nonly)
                summary.append(item)

                stem = f"{voice}_{i:02d}"
                sf.write(str(out / f"{stem}_native.wav"), wav.numpy(), SAMPLE_RATE)
                sf.write(
                    str(out / f"{stem}_override_cal.wav"),
                    audio_cal.detach().cpu().float().view(-1).numpy(),
                    SAMPLE_RATE,
                )
                sf.write(
                    str(out / f"{stem}_override_naive.wav"),
                    audio_naive.detach().cpu().float().view(-1).numpy(),
                    SAMPLE_RATE,
                )
                sf.write(
                    str(out / f"{stem}_override_raw.wav"),
                    audio_raw.detach().cpu().float().view(-1).numpy(),
                    SAMPLE_RATE,
                )
                sf.write(
                    str(out / f"{stem}_override_f0cal.wav"),
                    audio_f0only.detach().cpu().float().view(-1).numpy(),
                    SAMPLE_RATE,
                )
                sf.write(
                    str(out / f"{stem}_override_ncal.wav"),
                    audio_nonly.detach().cpu().float().view(-1).numpy(),
                    SAMPLE_RATE,
                )
                np.savez(
                    out / f"{stem}_curves.npz",
                    F0_kokoro=f0_k.numpy(),
                    N_kokoro=n_k.numpy(),
                    F0_yin_raw=f0_raw.numpy(),
                    E_yin_raw=n_raw.numpy(),
                    F0_yin_cal=f0_cal.numpy(),
                    F0_yin_cal_naive=f0_cal_naive.numpy(),
                    E_yin_cal=n_cal.numpy(),
                )

                # Plots
                fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex="col")
                t = np.arange(f0_k.numel())
                axes[0, 0].plot(t, f0_k.numpy(), label="Kokoro F0", color="#1f77b4", lw=1.5)
                axes[0, 0].plot(t, f0_cal.numpy(), label="YIN voiced-cal", color="#2ca02c", alpha=0.85)
                axes[0, 0].set_title(
                    f"F0 voiced-cal (corr all={item['corr_f0_cal_voiced_aware']:.3f}, "
                    f"both-v={item['corr_f0_both_voiced']:.3f})"
                )
                axes[0, 0].legend(loc="upper right", fontsize=8)
                axes[0, 0].set_ylabel("model units ≈ Hz")

                ax0b = axes[0, 1]
                ax0b.plot(t, f0_k.numpy(), label="Kokoro F0", color="#1f77b4", lw=1.5)
                ax0b2 = ax0b.twinx()
                ax0b2.plot(t, f0_raw.numpy(), label="YIN Hz raw", color="#ff7f0e", alpha=0.75)
                ax0b.set_title(
                    f"F0 raw (corr all={item['corr_f0_all']:.3f}, "
                    f"kok-v={item['corr_f0_kokoro_voiced']:.3f})"
                )
                ax0b.set_ylabel("Kokoro")
                ax0b2.set_ylabel("Hz")
                h1, l1 = ax0b.get_legend_handles_labels()
                h2, l2 = ax0b2.get_legend_handles_labels()
                ax0b.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

                axes[1, 0].plot(t[: n_k.numel()], n_k.numpy(), label="Kokoro N", color="#1f77b4", lw=1.5)
                axes[1, 0].plot(
                    t[: n_cal.numel()], n_cal.numpy(), label="E calibrated", color="#2ca02c", alpha=0.85
                )
                axes[1, 0].set_title(f"N calibrated (corr={item['corr_n_cal']:.3f})")
                axes[1, 0].legend(loc="upper right", fontsize=8)
                axes[1, 0].set_ylabel("model units")
                axes[1, 0].set_xlabel("frame")

                ax1b = axes[1, 1]
                ax1b.plot(t[: n_k.numel()], n_k.numpy(), label="Kokoro N", color="#1f77b4", lw=1.5)
                ax1b2 = ax1b.twinx()
                ax1b2.plot(t[: n_raw.numel()], n_raw.numpy(), label="log-E raw", color="#d62728", alpha=0.75)
                ax1b.set_title(f"N raw (corr={item['corr_n_raw']:.3f})")
                ax1b.set_ylabel("Kokoro")
                ax1b2.set_ylabel("log-energy")
                ax1b.set_xlabel("frame")
                h1, l1 = ax1b.get_legend_handles_labels()
                h2, l2 = ax1b2.get_legend_handles_labels()
                ax1b.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

                fig.suptitle(f"{voice} #{i}: {text[:70]}", fontsize=11)
                fig.tight_layout()
                png = out / f"{stem}_curves.png"
                fig.savefig(png, dpi=120)
                plt.close(fig)

                print(
                    f"{voice}#{i} T={item['T_f0']} "
                    f"F0 k mean={item['kokoro_F0']['mean']:.1f} "
                    f"yinHz={item['yin_raw_Hz']['mean']:.1f} | "
                    f"corr F0 all/kokV/bothV="
                    f"{item['corr_f0_all']:.3f}/"
                    f"{item['corr_f0_kokoro_voiced']:.3f}/"
                    f"{item['corr_f0_both_voiced']:.3f} "
                    f"calAware={item['corr_f0_cal_voiced_aware']:.3f} "
                    f"N={item['corr_n_cal']:.3f}",
                    flush=True,
                )

                esc = html_lib.escape
                html.append("<div class='card'>")
                html.append(f"<h3>#{i} {esc(voice)}</h3>")
                html.append(f"<p>{esc(text)}</p>")
                html.append(
                    "<table><tr><th>curve</th><th>corr</th><th>mean</th><th>std</th><th>notes</th></tr>"
                )
                html.append(
                    f"<tr><td>Kokoro F0</td><td>—</td>"
                    f"<td>{item['kokoro_F0']['mean']:.2f}</td>"
                    f"<td>{item['kokoro_F0']['std']:.2f}</td>"
                    f"<td>voiced frac={item['frac_kokoro_voiced']:.2f}</td></tr>"
                )
                html.append(
                    f"<tr><td>YIN raw (all frames)</td><td>{item['corr_f0_all']:.3f}</td>"
                    f"<td>{item['yin_raw_Hz']['mean']:.1f}</td>"
                    f"<td>{item['yin_raw_Hz']['std']:.1f}</td>"
                    f"<td>Hz; unvoiced noise hurts</td></tr>"
                )
                html.append(
                    f"<tr><td>YIN on Kokoro-voiced</td><td>{item['corr_f0_kokoro_voiced']:.3f}</td>"
                    f"<td>—</td><td>—</td><td>F0_k &gt; 50</td></tr>"
                )
                html.append(
                    f"<tr><td>YIN both-voiced</td><td>{item['corr_f0_both_voiced']:.3f}</td>"
                    f"<td>—</td><td>—</td><td>yin&gt;1 &amp; F0_k&gt;50</td></tr>"
                )
                html.append(
                    f"<tr><td>YIN voiced-cal</td><td>{item['corr_f0_cal_voiced_aware']:.3f}</td>"
                    f"<td>{item['yin_cal']['mean']:.2f}</td>"
                    f"<td>{item['yin_cal']['std']:.2f}</td>"
                    f"<td>voiced mean/std + zero UV</td></tr>"
                )
                html.append(
                    f"<tr><td>N / log-E cal</td><td>{item['corr_n_cal']:.3f}</td>"
                    f"<td>{item['energy_cal']['mean']:.2f}</td>"
                    f"<td>{item['energy_cal']['std']:.2f}</td>"
                    f"<td>global mean/std</td></tr>"
                )
                html.append("</table>")
                html.append(f"<p><img src='{png.name}' alt='curves'></p>")

                def audio_row(label: str, name: str, r: float) -> None:
                    html.append(
                        f"<p>{esc(label)} <small>rms={r:.4f}</small></p>"
                        f"<audio controls src='{name}'></audio>"
                    )

                audio_row("1) native Kokoro", f"{stem}_native.wav", item["rms_native"])
                audio_row(
                    "2) override YIN+E voiced-cal (recommended)",
                    f"{stem}_override_cal.wav",
                    item["rms_cal"],
                )
                audio_row(
                    "3) override YIN+E naive mean/std (no UV zeroing)",
                    f"{stem}_override_naive.wav",
                    item["rms_naive"],
                )
                audio_row(
                    "4) override YIN+E raw Hz (no calib)",
                    f"{stem}_override_raw.wav",
                    item["rms_raw"],
                )
                audio_row("5) override F0 voiced-cal only", f"{stem}_override_f0cal.wav", item["rms_f0only"])
                audio_row("6) override N cal only", f"{stem}_override_ncal.wav", item["rms_nonly"])
                html.append("</div>")

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    html.append("</body></html>")
    (out / "index.html").write_text("\n".join(html), encoding="utf-8")
    print(f"wrote {out / 'index.html'} and summary.json ({len(summary)} clips)", flush=True)


if __name__ == "__main__":
    main()
