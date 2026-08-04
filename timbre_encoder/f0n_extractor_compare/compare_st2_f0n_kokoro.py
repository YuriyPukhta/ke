"""
Compare Kokoro-native F0/N vs StyleTTS2 extractors on Kokoro-generated audio.

  F0 ← ST2 JDCNet pitch_extractor(mel)
  N  ← ST2 log_norm(mel)

Also reports YIN corr for reference. Listen: native / ST2 cal / ST2 raw / axes.

  .venv\\Scripts\\python.exe -m timbre_encoder.f0n_extractor_compare.compare_st2_f0n_kokoro
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
import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ST2_ROOT = ROOT / "third_party" / "StyleTTS2"
CKPT_DIR = ROOT / "timbre_encoder" / "checkpoints" / "styletts2_libritts" / "Models" / "LibriTTS"
_THIRD = ROOT / "third_party"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))
sys.path.insert(0, str(_THIRD))
sys.path.insert(0, str(ST2_ROOT))

from kokoro.pipeline import KPipeline  # noqa: E402
from kokoro_revers.model import KModel  # noqa: E402

from timbre_encoder.mel_preprocess import SAMPLE_RATE, waveform_to_mel  # noqa: E402
from timbre_encoder.optimize_speaker_ref_s import extract_f0_n, pack_slot  # noqa: E402
from timbre_encoder.st2_runtime import patch_torch_load  # noqa: E402

DEFAULT_TEXTS = [
    "Hello, this is a short test of pitch and energy.",
    "The quick brown fox jumps over the lazy dog near the river.",
    "I would like to know whether the extracted contours match Kokoro.",
]


def resize(x: torch.Tensor, T: int) -> torch.Tensor:
    x = x.float().view(1, 1, -1)
    return F.interpolate(x, size=T, mode="linear", align_corners=False).view(-1)


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
    """Voiced mean/std match; force 0 where source is unvoiced."""
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


def stats(x: torch.Tensor) -> dict[str, float]:
    t = x.float().detach().cpu().view(-1)
    return {
        "mean": float(t.mean()),
        "std": float(t.std()),
        "min": float(t.min()),
        "max": float(t.max()),
    }


def corr(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    x, y = a.float().view(-1), b.float().view(-1)
    if mask is not None:
        m = mask.bool().view(-1)
        if int(m.sum()) < 4:
            return float("nan")
        x, y = x[m], y[m]
    x, y = x - x.mean(), y - y.mean()
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
    for _, ps, _ in pipeline.en_tokenize(pipeline.g2p(text)[1]):
        return ps
    return None


def load_pitch_extractor(device: torch.device):
    patch_torch_load()
    from models import load_F0_models

    config = yaml.safe_load(open(CKPT_DIR / "config.yml", encoding="utf-8"))
    f0_path = ST2_ROOT / config["F0_path"]
    print(f"loading JDCNet from {f0_path}", flush=True)
    pe = load_F0_models(str(f0_path))
    pe.eval().to(device)
    for p in pe.parameters():
        p.requires_grad_(False)
    return pe


@torch.no_grad()
def extract_st2_f0_n(
    pitch_extractor,
    wave: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    wave [T] on any device → F0 (JDC), N (log_norm) on CPU 1D.
    Mel uses StyleTTS2 normalization (same as training).
    """
    from utils import log_norm

    mel = waveform_to_mel(wave.to(device), device=device)  # [1, 80, T]
    # JDC expects [B, 1, n_mels, T] in ST2 training path via mel.unsqueeze(1)
    F0_real, _, _ = pitch_extractor(mel.unsqueeze(1))
    N_real = log_norm(mel.unsqueeze(1)).squeeze(1)
    f0 = to_1d(F0_real).cpu()
    n = to_1d(N_real).cpu()
    return f0, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", type=str, default="af_heart,am_michael")
    ap.add_argument("--texts", type=str, nargs="*", default=None)
    ap.add_argument(
        "--out",
        type=str,
        default=str(HERE / "results" / "st2_f0n_kokoro_compare"),
    )
    args = ap.parse_args()

    texts = args.texts if args.texts else DEFAULT_TEXTS
    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} out={out}", flush=True)

    pitch_extractor = load_pitch_extractor(device)

    pipeline = KPipeline(lang_code="a", model=False)
    model = KModel(repo_id="hexgrad/Kokoro-82M").to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    summary: list[dict] = []
    html: list[str] = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Kokoro F0/N vs ST2 extractors</title>",
        "<style>body{font-family:sans-serif;max-width:1100px;margin:1.5rem auto;line-height:1.4}"
        "audio{width:100%;margin:.15rem 0}.card{margin:1.2rem 0;padding:1rem;border:1px solid #ccc}"
        "code{background:#f4f4f4;padding:0 .25rem}table{border-collapse:collapse;font-size:13px}"
        "td,th{border:1px solid #ddd;padding:.25rem .5rem;text-align:right}"
        "th:first-child,td:first-child{text-align:left}"
        "img{max-width:100%;border:1px solid #eee;margin:.5rem 0}</style></head><body>",
        "<h1>Kokoro native F0/N vs StyleTTS2 extractors</h1>",
        "<p><b>F0</b> = JDCNet <code>pitch_extractor(mel)</code>; "
        "<b>N</b> = <code>log_norm(mel)</code>. "
        "Same Kokoro-generated wav as source. "
        "YIN corr shown for comparison (no YIN override audio here).</p>"
        "<ul>"
        "<li><b>ST2 raw</b> — resize only</li>"
        "<li><b>ST2 cal</b> — F0 voiced mean/std + UV→0; N global mean/std</li>"
        "</ul>",
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
                    print(f"{voice}#{i} skip phonemes", flush=True)
                    continue
                input_ids = list(
                    filter(lambda j: j is not None, map(lambda p: model.vocab.get(p), phonemes))
                )
                if len(input_ids) != len(phonemes) or len(input_ids) + 2 > model.context_length:
                    print(f"{voice}#{i} skip tokens", flush=True)
                    continue

                ids = torch.LongTensor([[0, *input_ids, 0]]).to(device)
                ref_s = pack_slot(pack, len(phonemes)).to(device)

                audio_nat, pred_dur = model.forward_with_tokens(ids, ref_s, speed=1.0)
                aln = pred_dur_to_alignment(pred_dur, device)
                _, f0_k, n_k, _ = model.forward_with_alignment(ids, ref_s, aln)
                f0_k = to_1d(f0_k).cpu()
                n_k = to_1d(n_k).cpu()
                wav = audio_nat.detach().cpu().float().view(-1)

                f0_st2, n_st2 = extract_st2_f0_n(pitch_extractor, wav, device)
                f0_st2_r = resize(f0_st2, f0_k.numel())
                n_st2_r = resize(n_st2, n_k.numel())
                f0_st2_c = calibrate_f0_voiced(f0_st2, f0_k)
                n_st2_c = match_mean_std(n_st2, n_k)

                f0_yin, _ = extract_f0_n(wav.numpy(), sr=SAMPLE_RATE)
                f0_yin_r = resize(f0_yin, f0_k.numel())

                kok_v = f0_k > 50.0
                st2_v = f0_st2_r > 1.0
                both_v = kok_v & st2_v

                item = {
                    "voice": voice,
                    "i": i,
                    "text": text,
                    "T": int(f0_k.numel()),
                    "kokoro_F0": stats(f0_k),
                    "kokoro_N": stats(n_k),
                    "st2_F0": stats(f0_st2_r),
                    "st2_N": stats(n_st2_r),
                    "corr_f0_st2_all": corr(f0_st2_r, f0_k),
                    "corr_f0_st2_kokV": corr(f0_st2_r, f0_k, kok_v),
                    "corr_f0_st2_bothV": corr(f0_st2_r, f0_k, both_v),
                    "corr_f0_st2_cal": corr(f0_st2_c, f0_k),
                    "corr_f0_yin_kokV": corr(f0_yin_r, f0_k, kok_v),
                    "corr_n_st2": corr(n_st2_r, n_k),
                    "corr_n_st2_cal": corr(n_st2_c, n_k),
                    "rms_native": rms(wav),
                }

                audio_cal, _, _, _ = model.forward_with_alignment(
                    ids, ref_s, aln,
                    f0_override=f0_st2_c.to(device),
                    n_override=n_st2_c.to(device),
                )
                audio_raw, _, _, _ = model.forward_with_alignment(
                    ids, ref_s, aln,
                    f0_override=f0_st2_r.to(device),
                    n_override=n_st2_r.to(device),
                )
                audio_f0, _, _, _ = model.forward_with_alignment(
                    ids, ref_s, aln, f0_override=f0_st2_c.to(device)
                )
                audio_n, _, _, _ = model.forward_with_alignment(
                    ids, ref_s, aln, n_override=n_st2_c.to(device)
                )
                item["rms_cal"] = rms(audio_cal)
                item["rms_raw"] = rms(audio_raw)
                item["rms_f0"] = rms(audio_f0)
                item["rms_n"] = rms(audio_n)
                summary.append(item)

                stem = f"{voice}_{i:02d}"
                sf.write(str(out / f"{stem}_native.wav"), wav.numpy(), SAMPLE_RATE)
                for name, a in [
                    ("override_cal", audio_cal),
                    ("override_raw", audio_raw),
                    ("override_f0cal", audio_f0),
                    ("override_ncal", audio_n),
                ]:
                    sf.write(
                        str(out / f"{stem}_{name}.wav"),
                        a.detach().cpu().float().view(-1).numpy(),
                        SAMPLE_RATE,
                    )
                np.savez(
                    out / f"{stem}_curves.npz",
                    F0_kokoro=f0_k.numpy(),
                    N_kokoro=n_k.numpy(),
                    F0_st2=f0_st2_r.numpy(),
                    N_st2=n_st2_r.numpy(),
                    F0_st2_cal=f0_st2_c.numpy(),
                    N_st2_cal=n_st2_c.numpy(),
                    F0_yin=f0_yin_r.numpy(),
                )

                fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex="col")
                t = np.arange(f0_k.numel())
                axes[0, 0].plot(t, f0_k.numpy(), label="Kokoro F0", color="#1f77b4", lw=1.4)
                axes[0, 0].plot(t, f0_st2_c.numpy(), label="ST2 F0 cal", color="#2ca02c", alpha=0.85)
                axes[0, 0].set_title(
                    f"F0 cal (kokV corr ST2={item['corr_f0_st2_kokV']:.3f} "
                    f"YIN={item['corr_f0_yin_kokV']:.3f})"
                )
                axes[0, 0].legend(fontsize=8)
                axes[0, 0].set_ylabel("model units")

                axes[0, 1].plot(t, f0_k.numpy(), label="Kokoro", color="#1f77b4", lw=1.4)
                axes[0, 1].plot(t, f0_st2_r.numpy(), label="ST2 raw", color="#ff7f0e", alpha=0.8)
                axes[0, 1].plot(t, f0_yin_r.numpy(), label="YIN Hz", color="#9467bd", alpha=0.5)
                axes[0, 1].set_title(f"F0 raw (all corr ST2={item['corr_f0_st2_all']:.3f})")
                axes[0, 1].legend(fontsize=8)

                axes[1, 0].plot(t, n_k.numpy(), label="Kokoro N", color="#1f77b4", lw=1.4)
                axes[1, 0].plot(t, n_st2_c.numpy(), label="ST2 log_norm cal", color="#2ca02c", alpha=0.85)
                axes[1, 0].set_title(f"N cal (corr={item['corr_n_st2_cal']:.3f})")
                axes[1, 0].legend(fontsize=8)
                axes[1, 0].set_xlabel("frame")

                axes[1, 1].plot(t, n_k.numpy(), label="Kokoro N", color="#1f77b4", lw=1.4)
                axb = axes[1, 1].twinx()
                axb.plot(t, n_st2_r.numpy(), label="ST2 log_norm raw", color="#d62728", alpha=0.75)
                axes[1, 1].set_title(f"N raw (corr={item['corr_n_st2']:.3f})")
                h1, l1 = axes[1, 1].get_legend_handles_labels()
                h2, l2 = axb.get_legend_handles_labels()
                axes[1, 1].legend(h1 + h2, l1 + l2, fontsize=8)
                axes[1, 1].set_xlabel("frame")

                fig.suptitle(f"{voice} #{i}: {text[:70]}", fontsize=11)
                fig.tight_layout()
                png = out / f"{stem}_curves.png"
                fig.savefig(png, dpi=120)
                plt.close(fig)

                print(
                    f"{voice}#{i} T={item['T']} "
                    f"F0 ST2 kokV/YIN kokV="
                    f"{item['corr_f0_st2_kokV']:.3f}/{item['corr_f0_yin_kokV']:.3f} "
                    f"N ST2={item['corr_n_st2']:.3f}",
                    flush=True,
                )

                esc = html_lib.escape
                html.append("<div class='card'>")
                html.append(f"<h3>#{i} {esc(voice)}</h3><p>{esc(text)}</p>")
                html.append(
                    "<table><tr><th>curve</th><th>corr</th><th>mean</th><th>std</th></tr>"
                )
                html.append(
                    f"<tr><td>Kokoro F0</td><td>—</td>"
                    f"<td>{item['kokoro_F0']['mean']:.2f}</td>"
                    f"<td>{item['kokoro_F0']['std']:.2f}</td></tr>"
                )
                html.append(
                    f"<tr><td>ST2 F0 (all)</td><td>{item['corr_f0_st2_all']:.3f}</td>"
                    f"<td>{item['st2_F0']['mean']:.2f}</td>"
                    f"<td>{item['st2_F0']['std']:.2f}</td></tr>"
                )
                html.append(
                    f"<tr><td>ST2 F0 on Kokoro-voiced</td>"
                    f"<td><b>{item['corr_f0_st2_kokV']:.3f}</b></td><td>—</td><td>—</td></tr>"
                )
                html.append(
                    f"<tr><td>YIN F0 on Kokoro-voiced</td>"
                    f"<td>{item['corr_f0_yin_kokV']:.3f}</td><td>—</td><td>—</td></tr>"
                )
                html.append(
                    f"<tr><td>ST2 N / log_norm</td>"
                    f"<td><b>{item['corr_n_st2']:.3f}</b></td>"
                    f"<td>{item['st2_N']['mean']:.2f}</td>"
                    f"<td>{item['st2_N']['std']:.2f}</td></tr>"
                )
                html.append("</table>")
                html.append(f"<p><img src='{png.name}'></p>")

                def row(label: str, name: str, r: float) -> None:
                    html.append(
                        f"<p>{esc(label)} <small>rms={r:.4f}</small></p>"
                        f"<audio controls src='{name}'></audio>"
                    )

                row("1) native Kokoro", f"{stem}_native.wav", item["rms_native"])
                row("2) ST2 F0+N calibrated", f"{stem}_override_cal.wav", item["rms_cal"])
                row("3) ST2 F0+N raw", f"{stem}_override_raw.wav", item["rms_raw"])
                row("4) ST2 F0 cal only", f"{stem}_override_f0cal.wav", item["rms_f0"])
                row("5) ST2 N cal only", f"{stem}_override_ncal.wav", item["rms_n"])
                html.append("</div>")

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    html.append("</body></html>")
    (out / "index.html").write_text("\n".join(html), encoding="utf-8")
    print(f"wrote {out / 'index.html'} ({len(summary)} clips)", flush=True)


if __name__ == "__main__":
    main()
