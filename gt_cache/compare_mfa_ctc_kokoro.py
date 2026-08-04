"""Compare MFA vs wav2vec2-CTC durations against Kokoro duration_proj.

Uses LibriTTS-R train-clean-100:
  * MFA TextGrids from cdminix/libritts-r-aligned (ARPAbet phones/words)
  * CTC d_gt from kokoro_targets_v1 NPZs (Misaki phones)
  * Kokoro duration_proj on the same Misaki string (baseline voice)

Because MFA (ARPAbet) and Misaki inventories differ, phone-level metrics are
only for CTC↔Kokoro. Cross-inventory comparisons use:
  1) frame-progress correlation (which unit is active at each mel frame)
  2) word-boundary MAE when Misaki space-words match MFA word count
"""
from __future__ import annotations

import argparse
import csv
import json
import re
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

from gt_cache.cache_runpod_targets import allocate_with_min  # noqa: E402
from timbre_encoder.kokoro_embed_train.common import (  # noqa: E402
    DEFAULT_BASELINE,
    freeze_module,
    pack_slot,
    prepare_utt,
    prosody_heads_forward,
    to_1d,
)
from timbre_encoder.librispeech_dataset import parse_d_gt  # noqa: E402
from timbre_encoder.mel_preprocess import HOP_LENGTH, SAMPLE_RATE  # noqa: E402

FRAME_SEC = HOP_LENGTH / SAMPLE_RATE  # 12.5 ms
WORD_SEPS = set(" —–-\t")


def parse_textgrid(path: Path) -> dict[str, list[tuple[float, float, str]]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    tiers: dict[str, list[tuple[float, float, str]]] = {}
    for m in re.finditer(
        r'name = "([^"]+)"\s+xmin = ([0-9.]+)\s+xmax = ([0-9.]+)\s+'
        r"intervals: size = (\d+)\s+(.*?)\s*(?=item \[\d+\]:|^\Z)",
        text,
        flags=re.S | re.M,
    ):
        name = m.group(1)
        body = m.group(5)
        intervals = [
            (float(a), float(b), t)
            for a, b, t in re.findall(
                r'xmin = ([0-9.]+)\s+xmax = ([0-9.]+)\s+text = "([^"]*)"',
                body,
            )
        ]
        tiers[name] = intervals
    return tiers


def allocate_frames(raw_float: np.ndarray, total: int) -> list[int]:
    raw = np.maximum(np.asarray(raw_float, dtype=np.float64), 1.0)
    return allocate_with_min(raw, np.ones(len(raw), dtype=np.int64), total).astype(int).tolist()


def normalize_to_total(durs: list[int] | np.ndarray, total: int) -> list[int]:
    return allocate_frames(np.asarray(durs, dtype=np.float64), total)


def mfa_unit_frames(
    tiers: dict, mel_len: int, tier: str = "phones"
) -> tuple[list[str], list[int]]:
    units = [(a, b, t) for a, b, t in tiers.get(tier, []) if t.strip()]
    if not units:
        return [], []
    raw = np.asarray([(b - a) / FRAME_SEC for a, b, _ in units], dtype=np.float64)
    labels = [t for _, _, t in units]
    return labels, allocate_frames(raw, mel_len)


def group_misaki_words(phonemes: str, d_gt: list[int]) -> tuple[list[str], list[int]]:
    if len(phonemes) != len(d_gt):
        raise ValueError(f"len mismatch {len(phonemes)}!={len(d_gt)}")
    words: list[str] = []
    durs: list[int] = []
    cur_ph: list[str] = []
    cur_d = 0
    for ch, d in zip(phonemes, d_gt):
        if ch in WORD_SEPS:
            if cur_ph:
                words.append("".join(cur_ph))
                durs.append(cur_d + int(d))
                cur_ph, cur_d = [], 0
            elif durs:
                durs[-1] += int(d)
            continue
        # strip leading punctuation attached to first phone of a word
        if not cur_ph and ch in "\"“”‘’'`.,:;!?":
            if durs:
                durs[-1] += int(d)
            continue
        cur_ph.append(ch)
        cur_d += int(d)
    if cur_ph:
        words.append("".join(cur_ph))
        durs.append(cur_d)
    return words, durs


def progress_signal(durs: list[int], mel_len: int) -> np.ndarray:
    sig = np.zeros(mel_len, dtype=np.float64)
    pos = 0
    n = max(len(durs) - 1, 1)
    for i, d in enumerate(durs):
        end = min(mel_len, pos + int(d))
        if end > pos:
            sig[pos:end] = i / n
        pos = end
        if pos >= mel_len:
            break
    if pos < mel_len and durs:
        sig[pos:] = 1.0
    return sig


def metrics_vecs(ref: list[int], est: list[int]) -> dict[str, float]:
    a = np.asarray(ref, dtype=np.float64)
    b = np.asarray(est, dtype=np.float64)
    if len(a) < 2 or len(a) != len(b):
        return {"n": float(len(a)), "duration_mae": float("nan"), "boundary_mae": float("nan"), "corr": float("nan")}
    ba, bb = np.cumsum(a)[:-1], np.cumsum(b)[:-1]
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 and b.std() > 1e-9 else float("nan")
    return {
        "n": float(len(a)),
        "duration_mae": float(np.mean(np.abs(a - b))),
        "boundary_mae": float(np.mean(np.abs(ba - bb))),
        "corr": corr,
    }


def corr_1d(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 4 or a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = [k for k in rows[0].keys()]
    out: dict[str, float] = {"n_utts": float(len(rows))}
    for k in keys:
        vals = [r[k] for r in rows if r[k] == r[k]]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def pick_rows(manifest: Path, n: int, mfa_dir: Path, ctc_root: Path) -> list[dict]:
    out: list[dict] = []
    with manifest.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("align_status") != "ok":
                continue
            ap = Path(row["audio_path"])
            uid = ap.stem
            mfa = mfa_dir / f"{uid}.TextGrid"
            ctc = (
                ctc_root
                / "train-clean-100"
                / row["speaker_id"]
                / row["chapter_id"]
                / f"{uid}.npz"
            )
            if not mfa.is_file() or not ctc.is_file():
                continue
            ph = (row.get("phonemes_st2") or "").strip()
            if not ph or any(c in ph for c in ("?", "\ufffd", "\u2753")):
                continue
            d_tma = parse_d_gt(row.get("d_gt"))
            if not d_tma or len(d_tma) != len(ph):
                continue
            out.append({**row, "uid": uid, "mfa_path": str(mfa), "ctc_path": str(ctc)})
            if len(out) >= n:
                break
    return out


def plot_progress(path: Path, curves: dict[str, np.ndarray], title: str):
    fig, ax = plt.subplots(figsize=(9, 3.4))
    colors = {"kokoro": "#9467bd", "ctc": "#1f77b4", "mfa_phone": "#2ca02c", "mfa_word": "#ff7f0e"}
    for name, y in curves.items():
        ax.plot(y, label=name, color=colors.get(name), lw=1.5, alpha=0.9)
    ax.set_xlabel("mel frame")
    ax.set_ylabel("normalized unit progress")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_phone_bars(path: Path, phonemes: str, kokoro: list[int], ctc: list[int], title: str):
    labels = list(phonemes)
    n = min(len(labels), 60)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.22), 3.4))
    x = np.arange(n)
    w = 0.4
    ax.bar(x - w / 2, kokoro[:n], width=w, label="kokoro", color="#9467bd", alpha=0.85)
    ax.bar(x + w / 2, ctc[:n], width=w, label="ctc", color="#1f77b4", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels[:n], fontsize=7)
    ax.set_ylabel("frames")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "timbre_encoder/manifests/train-clean-100.filtered.csv",
    )
    ap.add_argument("--mfa-dir", type=Path, default=ROOT / "gt_cache/mfa")
    ap.add_argument("--ctc-root", type=Path, default=ROOT / "kokoro_targets_v1")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--baseline", type=str, default=DEFAULT_BASELINE)
    ap.add_argument("--plot-n", type=int, default=6)
    ap.add_argument("--out", type=Path, default=ROOT / "gt_cache/results/mfa_ctc_kokoro")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = pick_rows(args.manifest, args.n, args.mfa_dir, args.ctc_root)
    if not rows:
        raise SystemExit("no overlapping rows found")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} n={len(rows)}", flush=True)
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    model = KModel(repo_id="hexgrad/Kokoro-82M").to(device).eval()
    freeze_module(model)
    pack = pipeline.load_voice(args.baseline).to(device)

    phone_ctc_k: list[dict[str, float]] = []
    prog_ctc_k: list[dict[str, float]] = []
    prog_mfa_phone_k: list[dict[str, float]] = []
    prog_mfa_word_k: list[dict[str, float]] = []
    prog_mfa_phone_ctc: list[dict[str, float]] = []
    word_match_ctc_k: list[dict[str, float]] = []
    word_match_mfa_k: list[dict[str, float]] = []
    word_match_ctc_mfa: list[dict[str, float]] = []
    examples: list[dict] = []
    skipped = 0
    skip_reasons: dict[str, int] = {}

    def skip(reason: str) -> None:
        nonlocal skipped
        skipped += 1
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    for i, row in enumerate(rows):
        z = np.load(row["ctc_path"], allow_pickle=True)
        ph_ctc = str(z["phonemes"].item())
        d_ctc = [int(x) for x in z["d_gt"].tolist()]
        mel_len = int(z["mel_len"])
        if len(ph_ctc) != len(d_ctc):
            skip("ctc_len")
            continue
        if any(c in ph_ctc for c in ("?", "\ufffd", "\u2753")):
            skip("ctc_badchar")
            continue
        phonemes = ph_ctc

        prepared = prepare_utt(model, phonemes, d_ctc, device)
        if prepared is None:
            skip("prepare_utt")
            continue
        input_ids, aln, n_ph = prepared
        ref_s = pack_slot(pack, n_ph)
        try:
            pred_dur, _, _ = prosody_heads_forward(model, input_ids, ref_s, aln)
        except Exception:
            skip("prosody")
            continue
        pred = [float(x) for x in to_1d(pred_dur).cpu().tolist()]
        if len(pred) != n_ph + 2:
            skip("pred_len")
            continue
        d_k = list(pred[1:-1])
        d_k[0] += pred[0]
        d_k[-1] += pred[-1]
        if len(d_k) != len(d_ctc):
            skip("merge_len")
            continue
        d_k = normalize_to_total(d_k, mel_len)
        d_ctc_n = normalize_to_total(d_ctc, mel_len)

        phone_ctc_k.append(metrics_vecs(d_k, d_ctc_n))
        prog_k = progress_signal(d_k, mel_len)
        prog_ctc = progress_signal(d_ctc_n, mel_len)
        prog_ctc_k.append({"corr": corr_1d(prog_k, prog_ctc), "mae": float(np.mean(np.abs(prog_k - prog_ctc)))})

        tiers = parse_textgrid(Path(row["mfa_path"]))
        mfa_phones, mfa_phone_d = mfa_unit_frames(tiers, mel_len, "phones")
        mfa_words, mfa_word_d = mfa_unit_frames(tiers, mel_len, "words")
        if not mfa_phone_d:
            skip("no_mfa_phones")
            # still keep phone-level CTC metrics already appended
            phone_ctc_k.pop()
            prog_ctc_k.pop()
            continue
        prog_mfa_p = progress_signal(mfa_phone_d, mel_len)
        prog_mfa_w = progress_signal(mfa_word_d, mel_len) if mfa_word_d else None
        prog_mfa_phone_k.append(
            {"corr": corr_1d(prog_k, prog_mfa_p), "mae": float(np.mean(np.abs(prog_k - prog_mfa_p)))}
        )
        prog_mfa_phone_ctc.append(
            {"corr": corr_1d(prog_ctc, prog_mfa_p), "mae": float(np.mean(np.abs(prog_ctc - prog_mfa_p)))}
        )
        if prog_mfa_w is not None:
            prog_mfa_word_k.append(
                {"corr": corr_1d(prog_k, prog_mfa_w), "mae": float(np.mean(np.abs(prog_k - prog_mfa_w)))}
            )

        # Optional exact word-count match
        try:
            w_ph, w_ctc = group_misaki_words(phonemes, d_ctc_n)
            _, w_k = group_misaki_words(phonemes, d_k)
        except ValueError:
            w_ph, w_ctc, w_k = [], [], []
        w_nz = [(p, c, k) for p, c, k in zip(w_ph, w_ctc, w_k) if p]
        if w_nz and mfa_word_d and len(w_nz) == len(mfa_word_d):
            wc = normalize_to_total([c for _, c, _ in w_nz], mel_len)
            wk = normalize_to_total([k for _, _, k in w_nz], mel_len)
            wm = normalize_to_total(mfa_word_d, mel_len)
            word_match_ctc_k.append(metrics_vecs(wk, wc))
            word_match_mfa_k.append(metrics_vecs(wk, wm))
            word_match_ctc_mfa.append(metrics_vecs(wc, wm))

        if len(examples) < args.plot_n:
            curves = {"kokoro": prog_k, "ctc": prog_ctc, "mfa_phone": prog_mfa_p}
            if prog_mfa_w is not None:
                curves["mfa_word"] = prog_mfa_w
            png_p = args.out / f"ex_{len(examples):02d}_{row['uid']}_progress.png"
            png_b = args.out / f"ex_{len(examples):02d}_{row['uid']}_phones.png"
            plot_progress(png_p, curves, f"{row['uid']} alignment progress")
            plot_phone_bars(
                png_b,
                phonemes,
                d_k,
                d_ctc_n,
                f"{row['uid']} phone durs  corr={phone_ctc_k[-1]['corr']:.3f}",
            )
            examples.append(
                {
                    "uid": row["uid"],
                    "text": row.get("text", ""),
                    "progress_png": png_p.name,
                    "phones_png": png_b.name,
                    "phone_ctc_vs_kokoro": phone_ctc_k[-1],
                    "progress_ctc_vs_kokoro": prog_ctc_k[-1],
                    "progress_mfa_phone_vs_kokoro": prog_mfa_phone_k[-1],
                    "progress_mfa_phone_vs_ctc": prog_mfa_phone_ctc[-1],
                    "n_mfa_phones": len(mfa_phones),
                    "n_misaki": len(phonemes),
                    "n_mfa_words": len(mfa_words),
                }
            )

        if (i + 1) % 20 == 0:
            print(
                f"processed {i+1}/{len(rows)} kept={len(phone_ctc_k)} skipped={skipped}",
                flush=True,
            )

    summary = {
        "n_requested": args.n,
        "n_kept": len(phone_ctc_k),
        "n_skipped": skipped,
        "skip_reasons": skip_reasons,
        "n_word_count_matched": len(word_match_mfa_k),
        "baseline_voice": args.baseline,
        "frame_ms": 1000 * FRAME_SEC,
        "phone_level_ctc_vs_kokoro": mean_metrics(phone_ctc_k),
        "progress_ctc_vs_kokoro": mean_metrics(prog_ctc_k),
        "progress_mfa_phone_vs_kokoro": mean_metrics(prog_mfa_phone_k),
        "progress_mfa_word_vs_kokoro": mean_metrics(prog_mfa_word_k),
        "progress_mfa_phone_vs_ctc": mean_metrics(prog_mfa_phone_ctc),
        "word_matched_ctc_vs_kokoro": mean_metrics(word_match_ctc_k),
        "word_matched_mfa_vs_kokoro": mean_metrics(word_match_mfa_k),
        "word_matched_ctc_vs_mfa": mean_metrics(word_match_ctc_mfa),
        "examples": examples,
    }

    # Verdict
    ctc_corr = summary["progress_ctc_vs_kokoro"].get("corr", float("nan"))
    mfa_corr = summary["progress_mfa_phone_vs_kokoro"].get("corr", float("nan"))
    if ctc_corr == ctc_corr and mfa_corr == mfa_corr:
        better = "CTC" if ctc_corr >= mfa_corr else "MFA"
        summary["verdict"] = (
            f"{better} progress matches Kokoro better "
            f"(CTC corr={ctc_corr:.3f}, MFA-phone corr={mfa_corr:.3f})."
        )
    else:
        summary["verdict"] = "insufficient data"

    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    def fmt(m: dict, keys: tuple[str, ...]) -> str:
        parts = []
        for k in keys:
            v = m.get(k, float("nan"))
            parts.append(f"{k}={v:.3f}" if v == v else f"{k}=nan")
        return " · ".join(parts)

    html = [
        "<!doctype html><meta charset=utf-8>",
        "<title>MFA vs CTC vs Kokoro</title>",
        "<style>body{font-family:Segoe UI,sans-serif;margin:24px;max-width:1100px}"
        "table{border-collapse:collapse;margin:12px 0}td,th{border:1px solid #ccc;padding:6px 10px}"
        "img{max-width:100%;border:1px solid #ddd;margin:8px 0}</style>",
        "<h1>MFA vs wav2vec2-CTC vs Kokoro duration_proj</h1>",
        f"<p><b>{summary['verdict']}</b></p>",
        f"<p>kept={summary['n_kept']} skipped={skipped} "
        f"word-count-matched={summary['n_word_count_matched']} · voice={args.baseline}</p>",
        "<h2>Aggregate</h2><table>",
        "<tr><th>comparison</th><th>metrics</th></tr>",
        f"<tr><td>phone CTC ↔ Kokoro</td><td>{fmt(summary['phone_level_ctc_vs_kokoro'], ('corr','duration_mae','boundary_mae'))}</td></tr>",
        f"<tr><td>progress CTC ↔ Kokoro</td><td>{fmt(summary['progress_ctc_vs_kokoro'], ('corr','mae'))}</td></tr>",
        f"<tr><td>progress MFA-phone ↔ Kokoro</td><td>{fmt(summary['progress_mfa_phone_vs_kokoro'], ('corr','mae'))}</td></tr>",
        f"<tr><td>progress MFA-word ↔ Kokoro</td><td>{fmt(summary['progress_mfa_word_vs_kokoro'], ('corr','mae'))}</td></tr>",
        f"<tr><td>progress MFA-phone ↔ CTC</td><td>{fmt(summary['progress_mfa_phone_vs_ctc'], ('corr','mae'))}</td></tr>",
        f"<tr><td>word-matched MFA ↔ Kokoro</td><td>{fmt(summary['word_matched_mfa_vs_kokoro'], ('corr','duration_mae','boundary_mae'))}</td></tr>",
        f"<tr><td>word-matched CTC ↔ Kokoro</td><td>{fmt(summary['word_matched_ctc_vs_kokoro'], ('corr','duration_mae','boundary_mae'))}</td></tr>",
        "</table><h2>Examples</h2>",
    ]
    for ex in examples:
        html += [
            f"<h3>{ex['uid']}</h3>",
            f"<p>{ex['text']}</p>",
            "<p>"
            f"phone CTC↔Kokoro corr={ex['phone_ctc_vs_kokoro']['corr']:.3f} · "
            f"progress CTC↔Kokoro={ex['progress_ctc_vs_kokoro']['corr']:.3f} · "
            f"progress MFA↔Kokoro={ex['progress_mfa_phone_vs_kokoro']['corr']:.3f}"
            "</p>",
            f"<img src='{ex['progress_png']}'>",
            f"<img src='{ex['phones_png']}'>",
        ]
    (args.out / "index.html").write_text("\n".join(html), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "examples"}, ensure_ascii=True, indent=2), flush=True)
    print(f"wrote {args.out / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
