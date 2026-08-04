"""
Test whether d_gt lengths work with Kokoro LSTM duration predictor.

Picks short samples from several LibriTTS speakers, uses phonemes_st2 (or
normalizes on the fly), runs Kokoro ProsodyPredictor LSTM, compares pred_dur
vs d_gt, and synthesizes LSTM-predicted vs forced-d_gt audio.

Usage:
  .venv\\Scripts\\python.exe -m timbre_encoder.test_d_gt_lstm
  .venv\\Scripts\\python.exe -m timbre_encoder.test_d_gt_lstm --speakers 4 --max-phonemes 80
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

from kokoro.pipeline import KPipeline  # noqa: E402

from timbre_encoder.aligner import FrozenTextAligner  # noqa: E402
from timbre_encoder.librispeech_dataset import (  # noqa: E402
    parse_d_gt,
    text_to_kokoro_phonemes,
)
from timbre_encoder.mel_preprocess import SAMPLE_RATE, waveform_to_mel  # noqa: E402
from timbre_encoder.phoneme_norm import (  # noqa: E402
    normalize_kokoro_phonemes_for_st2,
    norm_summary,
)


def pick_short_samples(csv_path: Path, n_speakers: int, min_chars: int = 12, max_chars: int = 90):
    """Pick one short-but-not-tiny utterance per speaker."""
    by_spk: dict[str, list] = {}
    with csv_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            text = (row.get("text") or "").strip()
            if not text or not (min_chars <= len(text) <= max_chars):
                continue
            # Prefer rows that already have phonemes_st2 + d_gt
            by_spk.setdefault(row["speaker_id"], []).append(row)

    picks = []
    for spk in sorted(by_spk.keys(), key=lambda s: int(s) if s.isdigit() else s):
        candidates = sorted(
            by_spk[spk],
            key=lambda r: (
                0 if (r.get("phonemes_st2") and r.get("align_status") == "ok") else 1,
                len(r.get("text") or ""),
            ),
        )
        picks.append(candidates[0])
        if len(picks) >= n_speakers:
            break
    return picks


def kokoro_ids(model, phonemes: str, device: torch.device):
    dropped = []
    ids = []
    for p in phonemes:
        i = model.vocab.get(p)
        if i is None:
            dropped.append(p)
        else:
            ids.append(i)
    input_ids = torch.LongTensor([[0, *ids, 0]]).to(device)
    return input_ids, ids, dropped


def d_gt_to_target_dur(d_gt: list[int], device: torch.device) -> torch.Tensor:
    d_list = [1] + [max(1, int(x)) for x in d_gt] + [1]
    return torch.tensor(d_list, dtype=torch.long, device=device)


def load_ref_wav(audio_path: Path, max_sec: float = 8.0) -> np.ndarray:
    wav, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    wav, _ = librosa.effects.trim(wav, top_db=30)
    max_samples = int(max_sec * SAMPLE_RATE)
    if wav.shape[0] > max_samples:
        wav = wav[:max_samples]
    return wav


def resolve_phonemes(row: dict, pipeline) -> tuple[str, str, bool, int, str]:
    """Return phonemes_raw, phonemes_st2, norm_applied, delta_len, norm_ops."""
    if (row.get("phonemes_st2") or "").strip() and row.get("norm_applied") in ("0", "1"):
        return (
            row["phonemes"],
            row["phonemes_st2"],
            row["norm_applied"] == "1",
            int(row.get("delta_len") or 0),
            row.get("norm_ops") or "",
        )

    legacy_raw = (row.get("phonemes_raw") or "").strip()
    if legacy_raw:
        raw = legacy_raw
    elif (row.get("phonemes") or "").strip():
        raw = row["phonemes"]
    else:
        raw = text_to_kokoro_phonemes(row["text"], pipeline)

    nr = normalize_kokoro_phonemes_for_st2(raw)
    return nr.raw, nr.normalized, nr.changed, nr.delta_len, norm_summary(nr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=str,
        default=str(ROOT / "timbre_encoder/manifests/dev-clean.filtered.csv"),
    )
    ap.add_argument("--speakers", type=int, default=4)
    ap.add_argument("--max-phonemes", type=int, default=80)
    ap.add_argument("--min-phonemes", type=int, default=8)
    ap.add_argument("--voice", type=str, default="af_heart")
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "timbre_encoder/runs/d_gt_lstm_test"),
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}", flush=True)
    print("Loading Kokoro + ST2 aligner...", flush=True)
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")

    from demo.kokoro_revers.model import KModel as KModelRev

    krev = KModelRev(repo_id="hexgrad/Kokoro-82M").to(device).eval()
    for p in krev.parameters():
        p.requires_grad_(False)

    aligner = FrozenTextAligner(device=device)
    voice_pack = pipeline.load_voice(args.voice).to(device)

    picks = pick_short_samples(Path(args.csv), args.speakers, min_chars=12, max_chars=90)
    print(f"picked {len(picks)} speakers from {args.csv}", flush=True)

    report_lines = []
    esc = html_lib.escape
    html_parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>d_gt vs Kokoro LSTM</title>",
        "<style>body{font-family:sans-serif;max-width:960px;margin:24px auto;padding:0 16px}",
        "table{border-collapse:collapse;width:100%;font-size:13px}",
        "td,th{border:1px solid #ccc;padding:4px 8px;text-align:left}",
        "code{font-size:12px;word-break:break-all}.ok{color:green}.bad{color:#b00}</style></head><body>",
        "<h1>d_gt length vs Kokoro LSTM</h1>",
        f"<p>Voice pack: <b>{esc(args.voice)}</b>. "
        "Forced audio uses <code>target_dur = [1]+d_gt+[1]</code> (BOS/EOS).</p>",
    ]

    for i, row in enumerate(picks, 1):
        spk = row["speaker_id"]
        text = row["text"]
        audio_path = Path(row["audio_path"])
        print(f"\n=== [{i}] speaker={spk} ===", flush=True)
        print(f"text: {text}", flush=True)

        phonemes_raw, phonemes_st2, nr_changed, delta_len, ops = resolve_phonemes(row, pipeline)

        if not (args.min_phonemes <= len(phonemes_st2) <= args.max_phonemes):
            print(f"  skip: st2 len={len(phonemes_st2)} outside range", flush=True)
            continue

        cached_d = parse_d_gt(row.get("d_gt"))
        if (
            cached_d is not None
            and len(cached_d) == len(phonemes_st2)
            and (row.get("phonemes_st2") or "").strip()
        ):
            d_gt_list = cached_d
            d_src = "cache"
        else:
            wav = load_ref_wav(audio_path)
            mel = waveform_to_mel(torch.from_numpy(wav.astype(np.float32)), device=device).squeeze(0)
            _, d_gt_t, _ = aligner.align(mel, phonemes_st2)
            d_gt_list = [max(1, int(round(float(x)))) for x in d_gt_t.tolist()]
            d_src = "computed"

        print(
            f"  raw={len(phonemes_raw)} st2={len(phonemes_st2)} "
            f"norm={nr_changed} delta={delta_len} ops={ops}",
            flush=True,
        )
        print(f"  d_gt source={d_src} len={len(d_gt_list)} sum={sum(d_gt_list)}", flush=True)

        input_ids, ids, dropped = kokoro_ids(krev, phonemes_st2, device)
        n_tok = int(input_ids.shape[1])
        length_ok = len(ids) == len(phonemes_st2) == len(d_gt_list)
        print(
            f"  Kokoro ids N={len(ids)} N+2={n_tok} dropped={dropped!r} length_ok={length_ok}",
            flush=True,
        )

        pack_idx = min(max(len(phonemes_st2) - 1, 0), voice_pack.shape[0] - 1)
        ref_s = voice_pack[pack_idx]
        if ref_s.dim() == 1:
            ref_s = ref_s.unsqueeze(0)

        with torch.no_grad():
            audio_lstm, pred_dur = krev.forward_with_tokens(input_ids, ref_s, speed=1)
        if pred_dur.dim() == 0:
            pred_dur = pred_dur.unsqueeze(0)
        pred_list = [int(x) for x in pred_dur.tolist()]
        pred_core = pred_list[1:-1]

        target_dur = d_gt_to_target_dur(d_gt_list, device)
        length_fit = int(target_dur.shape[0]) == n_tok
        print(
            f"  pred_dur={len(pred_list)} target_dur={int(target_dur.shape[0])} fit={length_fit}",
            flush=True,
        )

        audio_forced = None
        forced_ok = False
        forced_err = ""
        if length_fit:
            try:
                with torch.no_grad():
                    audio_forced, _ = krev.forward_with_tokens_grad(
                        input_ids, ref_s, speed=1, target_dur=target_dur
                    )
                forced_ok = True
            except Exception as e:
                forced_err = f"{type(e).__name__}: {e}"
                print(f"  forced FAIL: {forced_err}", flush=True)
        else:
            forced_err = (
                f"length mismatch: target_dur={int(target_dur.shape[0])} input_ids={n_tok}"
            )
            print(f"  forced SKIP: {forced_err}", flush=True)

        if length_ok and len(pred_core) == len(d_gt_list):
            pred_t = torch.tensor(pred_core, dtype=torch.float)
            gt_t = torch.tensor(d_gt_list, dtype=torch.float)
            mae = float((pred_t - gt_t).abs().mean())
            sum_pred, sum_gt = int(pred_t.sum()), int(gt_t.sum())
        else:
            mae = float("nan")
            sum_pred, sum_gt = sum(pred_core), sum(d_gt_list)

        print(f"  MAE={mae:.3f} sum_pred={sum_pred} sum_gt={sum_gt}", flush=True)
        print(f"  pred[:20]={pred_core[:20]}", flush=True)
        print(f"  d_gt[:20]={d_gt_list[:20]}", flush=True)

        stem = f"{i:02d}_spk{spk}"
        ref_path = out_dir / f"{stem}_ref.wav"
        lstm_path = out_dir / f"{stem}_lstm_pred.wav"
        forced_path = out_dir / f"{stem}_forced_dgt.wav"
        sf.write(str(ref_path), load_ref_wav(audio_path), SAMPLE_RATE)
        sf.write(str(lstm_path), audio_lstm.detach().cpu().numpy(), SAMPLE_RATE)
        if forced_ok and audio_forced is not None:
            sf.write(str(forced_path), audio_forced.detach().cpu().numpy(), SAMPLE_RATE)

        length_cls = "ok" if length_ok and length_fit and forced_ok else "bad"
        report_lines.append(
            f"spk={spk} | raw={len(phonemes_raw)} st2={len(phonemes_st2)} "
            f"d_gt={len(d_gt_list)} ids={len(ids)} N+2={n_tok} | "
            f"match={length_ok} fit={length_fit} forced={forced_ok} | "
            f"MAE={mae:.3f} sum_pred={sum_pred} sum_gt={sum_gt} | {ops}"
        )

        html_parts += [
            f"<h2>#{i} speaker {esc(spk)}</h2>",
            f"<p>{esc(text)}</p>",
            "<table>",
            f"<tr><th>phonemes (raw)</th><td><code>{esc(phonemes_raw)}</code> (len={len(phonemes_raw)})</td></tr>",
            f"<tr><th>phonemes_st2</th><td><code>{esc(phonemes_st2)}</code> (len={len(phonemes_st2)})</td></tr>",
            f"<tr><th>norm</th><td>applied={nr_changed} delta_len={delta_len} ops={esc(ops)}</td></tr>",
            f"<tr><th>d_gt</th><td>source={d_src} len={len(d_gt_list)} sum={sum(d_gt_list)} <code>{d_gt_list}</code></td></tr>",
            f"<tr><th>Kokoro encode</th><td>ids={len(ids)} N+2={n_tok} dropped={esc(repr(dropped))}</td></tr>",
            f"<tr><th>LSTM pred_dur</th><td>len={len(pred_list)} core_sum={sum_pred} <code>{pred_core}</code></td></tr>",
            f"<tr><th>Compare</th><td class='{length_cls}'>"
            f"ids==st2==d_gt: {length_ok} | target_dur fit: {length_fit} | "
            f"forced synth: {forced_ok} | MAE={mae:.3f}</td></tr>",
        ]
        if forced_err:
            html_parts.append(
                f"<tr><th>forced error</th><td class='bad'>{esc(forced_err)}</td></tr>"
            )
        html_parts += [
            "</table>",
            "<p><b>Reference (LibriTTS)</b></p>",
            f"<audio controls src='{ref_path.name}'></audio>",
            f"<p><b>Kokoro LSTM pred_dur</b> (voice={esc(args.voice)})</p>",
            f"<audio controls src='{lstm_path.name}'></audio>",
        ]
        if forced_ok:
            html_parts += [
                "<p><b>Kokoro forced d_gt</b></p>",
                f"<audio controls src='{forced_path.name}'></audio>",
            ]
        html_parts.append("<hr>")

        (out_dir / f"{stem}_meta.txt").write_text(
            "\n".join(
                [
                    f"speaker={spk}",
                    f"text={text}",
                    f"phonemes={phonemes_raw}",
                    f"phonemes_st2={phonemes_st2}",
                    f"norm_applied={nr_changed}",
                    f"delta_len={delta_len}",
                    f"norm_ops={ops}",
                    f"d_gt_source={d_src}",
                    f"d_gt={d_gt_list}",
                    f"pred_dur={pred_list}",
                    f"pred_core={pred_core}",
                    f"kokoro_ids_len={len(ids)}",
                    f"dropped={dropped!r}",
                    f"length_ok={length_ok}",
                    f"length_fit={length_fit}",
                    f"forced_ok={forced_ok}",
                    f"forced_err={forced_err}",
                    f"mae={mae}",
                    f"sum_pred={sum_pred}",
                    f"sum_gt={sum_gt}",
                ]
            ),
            encoding="utf-8",
        )

    html_parts += [
        "<h2>Summary</h2><pre>",
        esc("\n".join(report_lines)),
        "</pre></body></html>",
    ]
    index = out_dir / "index.html"
    index.write_text("\n".join(html_parts), encoding="utf-8")
    (out_dir / "summary.txt").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nWrote {index}", flush=True)
    for line in report_lines:
        print(line, flush=True)


if __name__ == "__main__":
    main()
