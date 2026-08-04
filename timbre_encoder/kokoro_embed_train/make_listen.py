"""
Listen HTML after stage1/2: original | style+baseline | full vector.

  .venv\\Scripts\\python.exe -m timbre_encoder.kokoro_embed_train.make_listen \\
      --ckpt timbre_encoder/runs/kokoro_embed/stage1/stage1_final.pt \\
      --manifest timbre_encoder/manifests/dev-clean.filtered.csv --limit 8
"""
from __future__ import annotations

import argparse
import csv
import html as html_lib
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

from kokoro.pipeline import KPipeline  # noqa: E402
from kokoro_revers.model import KModel  # noqa: E402

from timbre_encoder.dual_encoder import DualStyleEncoder, crop_mel_for_style  # noqa: E402
from gt_cache import feature_path, load_curve  # noqa: E402
from gt_cache.utils import DEFAULT_FEATURES_ROOT  # noqa: E402
from timbre_encoder.kokoro_embed_train.common import (  # noqa: E402
    DEFAULT_BASELINE,
    MAX_WAV_SEC,
    calibrate_st2_to_kokoro,
    freeze_module,
    kokoro_f0n_for_ref,
    load_ckpt,
    pack_slot,
    prepare_utt,
)
from timbre_encoder.librispeech_dataset import parse_d_gt  # noqa: E402
from timbre_encoder.mel_preprocess import SAMPLE_RATE, waveform_to_mel  # noqa: E402


def load_wav(path: str, max_sec: float = MAX_WAV_SEC) -> np.ndarray:
    wav, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    wav, _ = librosa.effects.trim(wav, top_db=30)
    max_samples = int(max_sec * SAMPLE_RATE)
    if wav.shape[0] > max_samples:
        wav = wav[:max_samples]
    return wav.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--features-root", type=str, default=str(DEFAULT_FEATURES_ROOT))
    ap.add_argument("--baseline", type=str, default="")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--no-f0n-override", action="store_true")
    ap.add_argument("--timbre-only", action="store_true", help="Force style+baseline only for full")
    ap.add_argument(
        "--speaker",
        type=str,
        default="",
        help="Comma-separated speaker_id(s); empty = diverse pick",
    )
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = load_ckpt(args.ckpt, map_location=device)
    baseline_name = args.baseline or ckpt.get("baseline") or DEFAULT_BASELINE
    out = Path(args.out) if args.out else HERE / "results" / "listen" / Path(args.ckpt).stem
    out.mkdir(parents=True, exist_ok=True)
    features_root = Path(args.features_root)
    allowed = {s.strip() for s in args.speaker.split(",") if s.strip()} or None

    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    model = KModel(repo_id="hexgrad/Kokoro-82M").to(device)
    freeze_module(model)
    baseline_pack = pipeline.load_voice(baseline_name).to(device)

    encoders = DualStyleEncoder().to(device)
    encoders.load_state_dict(ckpt["encoders"])
    if "residual_scale" in ckpt:
        encoders.residual_scale = float(ckpt["residual_scale"])
    encoders.eval()

    rows = []
    with open(args.manifest, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("align_status") != "ok":
                continue
            if allowed is not None and str(row.get("speaker_id", "")) not in allowed:
                continue
            rows.append(row)

    cards = []
    n_done = 0
    seen_spk: set[str] = set()
    pool = list(rows)
    picked: list[dict] = []
    # Single-speaker: just take first N; multi: prefer diverse speakers
    if allowed is not None and len(allowed) == 1:
        picked = pool[: args.limit]
    else:
        while len(picked) < args.limit * 2 and pool:
            progressed = False
            for i, row in enumerate(list(pool)):
                spk = row["speaker_id"]
                if spk not in seen_spk or len(seen_spk) >= args.limit:
                    picked.append(row)
                    seen_spk.add(spk)
                    pool.pop(i)
                    progressed = True
                    break
            if not progressed:
                break
        if len(picked) < args.limit:
            picked.extend(pool[: args.limit - len(picked)])

    for row in picked:
        if n_done >= args.limit:
            break
        ph = (row.get("phonemes_st2") or "").strip()
        d_gt = parse_d_gt(row.get("d_gt"))
        if not ph or not d_gt:
            continue
        prep = prepare_utt(model, ph, d_gt, device)
        if prep is None:
            continue
        input_ids, aln, n_ph = prep
        try:
            wav = load_wav(row["audio_path"])
            wave_t = torch.from_numpy(wav)
            mel = waveform_to_mel(wave_t, device=device).squeeze(0)
            base_ref = pack_slot(baseline_pack, n_ph)
            mel_crop = crop_mel_for_style(mel.unsqueeze(0), max_frames=192, min_frames=80)
            with torch.no_grad():
                s_a, s_p_full = encoders(
                    mel_crop, base_ref, timbre_only=False if not args.timbre_only else True
                )
                s_a_tb, s_p_tb = encoders(mel_crop, base_ref, timbre_only=True)
            ref_style_base = torch.cat([s_a_tb, s_p_tb], dim=-1)
            ref_full = torch.cat([s_a, s_p_full], dim=-1)

            f0_ov = n_ov = None
            if not args.no_f0n_override:
                f0_st2 = load_curve(feature_path(row["audio_path"], "f0", features_root))
                n_st2 = load_curve(feature_path(row["audio_path"], "n", features_root))
                if f0_st2 is not None and n_st2 is not None:
                    f0_ref, n_ref = kokoro_f0n_for_ref(model, input_ids, base_ref, aln)
                    f0_ov, n_ov = calibrate_st2_to_kokoro(
                        f0_st2.to(device), n_st2.to(device), f0_ref, n_ref
                    )

            stem = f"{n_done:02d}_{row['speaker_id']}"
            sf.write(str(out / f"{stem}_0_original.wav"), wav, SAMPLE_RATE)

            a1, _, _, _ = model.forward_with_alignment(
                input_ids, ref_style_base, aln, f0_override=f0_ov, n_override=n_ov
            )
            sf.write(
                str(out / f"{stem}_1_style_baseline.wav"),
                a1.detach().cpu().numpy().reshape(-1),
                SAMPLE_RATE,
            )
            a2, _, _, _ = model.forward_with_alignment(
                input_ids, ref_full, aln, f0_override=f0_ov, n_override=n_ov
            )
            sf.write(
                str(out / f"{stem}_2_full_vector.wav"),
                a2.detach().cpu().numpy().reshape(-1),
                SAMPLE_RATE,
            )

            cards.append(
                f"<div class='card'><h3>{html_lib.escape(stem)}</h3>"
                f"<p>{html_lib.escape((row.get('text') or '')[:160])}</p>"
                f"<p><b>original</b></p><audio controls src='{stem}_0_original.wav'></audio>"
                f"<p><b>style + baseline</b></p><audio controls src='{stem}_1_style_baseline.wav'></audio>"
                f"<p><b>full vector</b></p><audio controls src='{stem}_2_full_vector.wav'></audio>"
                f"</div>"
            )
            n_done += 1
            print(f"wrote {stem}", flush=True)
        except Exception as e:
            print(f"skip: {e}", flush=True)
            continue

    html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<title>Kokoro embed listen</title>
<style>body{{font-family:sans-serif;max-width:900px;margin:1.5rem auto}}
.card{{margin:1rem 0;padding:1rem;border:1px solid #ccc}}audio{{width:100%}}</style>
</head><body>
<h1>Listen: original | style+baseline | full</h1>
<p>ckpt={html_lib.escape(args.ckpt)} baseline={html_lib.escape(baseline_name)}</p>
{''.join(cards)}
</body></html>"""
    (out / "index.html").write_text(html, encoding="utf-8")
    print(f"done n={n_done} -> {out / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
