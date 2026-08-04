"""
Compare per-phoneme duration sequences: d_gt (ST2-norm) vs Kokoro LSTM pred.

Usage:
  .venv\\Scripts\\python.exe -m timbre_encoder.analyze_duration_similarity
"""
from __future__ import annotations

import argparse
import ast
import html as html_lib
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_meta(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        data[k.strip()] = v.strip()
    return data


def parse_int_list(s: str) -> list[int]:
    return list(ast.literal_eval(s))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    def rank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(x), dtype=float)
        return ranks

    return pearson(rank(a), rank(b))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def sparkline(vals: list[int]) -> str:
    blocks = " ▁▂▃▄▅▆▇█"
    mx = max(vals) or 1
    return "".join(blocks[int(round((v / mx) * (len(blocks) - 1)))] for v in vals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dir",
        type=str,
        default=str(ROOT / "timbre_encoder/runs/d_gt_lstm_test"),
    )
    args = ap.parse_args()
    out_dir = Path(args.dir)
    metas = sorted(out_dir.glob("*_meta.txt"))
    if not metas:
        raise SystemExit(f"no meta files in {out_dir}")

    esc = html_lib.escape
    lines: list[str] = []
    summary_rows: list[dict] = []
    detail_html: list[str] = []

    for meta_path in metas:
        m = parse_meta(meta_path)
        spk = m.get("speaker", "?")
        text = m.get("text", "")
        phonemes_st2 = m.get("phonemes_st2", "")
        d_gt = parse_int_list(m["d_gt"])
        pred_core = parse_int_list(m["pred_core"])
        if len(d_gt) != len(pred_core):
            lines.append(f"spk={spk} SKIP length mismatch d_gt={len(d_gt)} pred={len(pred_core)}")
            continue
        if len(phonemes_st2) != len(d_gt):
            # still compare vectors; phone labels may be truncated in display
            pass

        a = np.array(d_gt, dtype=float)
        b = np.array(pred_core, dtype=float)
        mae = float(np.mean(np.abs(a - b)))
        rmse = float(np.sqrt(np.mean((a - b) ** 2)))
        r_p = pearson(a, b)
        r_s = spearman(a, b)
        cos = cosine(a, b)
        sum_gt, sum_pred = int(a.sum()), int(b.sum())
        ratio = sum_pred / sum_gt if sum_gt else float("nan")
        within1 = float(np.mean(np.abs(a - b) <= 1))
        within2 = float(np.mean(np.abs(a - b) <= 2))
        peak_gt = int(np.argmax(a))
        peak_pred = int(np.argmax(b))

        summary_rows.append(
            dict(
                spk=spk,
                n=len(d_gt),
                sum_gt=sum_gt,
                sum_pred=sum_pred,
                ratio=ratio,
                mae=mae,
                rmse=rmse,
                pearson=r_p,
                spearman=r_s,
                cosine=cos,
                within1=within1,
                within2=within2,
                peak_gt=peak_gt,
                peak_pred=peak_pred,
                norm=m.get("norm_ops", ""),
            )
        )

        lines.append(
            f"spk={spk} N={len(d_gt)} | sum_gt={sum_gt} sum_pred={sum_pred} "
            f"ratio={ratio:.3f} | MAE={mae:.3f} RMSE={rmse:.3f} | "
            f"pearson={r_p:.3f} spearman={r_s:.3f} cos={cos:.3f} | "
            f"within1={within1:.0%} within2={within2:.0%} | "
            f"peak_gt@{peak_gt}('{phonemes_st2[peak_gt] if peak_gt < len(phonemes_st2) else '?'}') "
            f"peak_pred@{peak_pred}('{phonemes_st2[peak_pred] if peak_pred < len(phonemes_st2) else '?'}') | "
            f"{m.get('norm_ops', '')}"
        )
        lines.append(f"  text: {text}")
        lines.append(f"  d_gt  {sparkline(d_gt)}")
        lines.append(f"  pred  {sparkline(pred_core)}")
        lines.append(f"  d_gt  = {d_gt}")
        lines.append(f"  pred  = {pred_core}")
        lines.append("")

        detail_html += [
            f"<h2>Speaker {esc(spk)}</h2>",
            f"<p>{esc(text)}</p>",
            f"<p><code>phonemes_st2</code> len={len(phonemes_st2)}: "
            f"<code>{esc(phonemes_st2)}</code></p>",
            "<table><tr>"
            "<th>sum_gt</th><th>sum_pred</th><th>ratio</th>"
            "<th>MAE</th><th>Pearson</th><th>Spearman</th><th>cosine</th>"
            "<th>|Δ|≤1</th><th>|Δ|≤2</th></tr>",
            f"<tr><td>{sum_gt}</td><td>{sum_pred}</td><td>{ratio:.3f}</td>"
            f"<td>{mae:.3f}</td><td>{r_p:.3f}</td><td>{r_s:.3f}</td><td>{cos:.3f}</td>"
            f"<td>{within1:.0%}</td><td>{within2:.0%}</td></tr></table>",
            f"<p class='mono'>d_gt  {esc(sparkline(d_gt))}<br>"
            f"pred  {esc(sparkline(pred_core))}</p>",
            "<table><tr><th>#</th><th class='ph'>phone</th><th>d_gt</th>"
            "<th>pred</th><th>Δ</th></tr>",
        ]
        phones = list(phonemes_st2) if len(phonemes_st2) == len(d_gt) else ["?"] * len(d_gt)
        for i, (ch, dg, pr) in enumerate(zip(phones, d_gt, pred_core)):
            diff = pr - dg
            cls = "ok" if abs(diff) <= 1 else ("warn" if abs(diff) <= 2 else "")
            detail_html.append(
                f"<tr class='{cls}'><td>{i}</td><td class='ph'><code>{esc(ch)}</code></td>"
                f"<td>{dg}</td><td>{pr}</td><td>{diff:+d}</td></tr>"
            )
        detail_html.append("</table><hr>")

    ratios = [s["ratio"] for s in summary_rows]
    pearsons = [s["pearson"] for s in summary_rows if not math.isnan(s["pearson"])]
    header = [
        "=== Duration sequence similarity: d_gt (phonemes_st2) vs Kokoro LSTM ===",
        f"speakers={len(summary_rows)} | mean ratio pred/gt={float(np.mean(ratios)):.3f} "
        f"| mean pearson={float(np.mean(pearsons)):.3f}" if pearsons else f"speakers={len(summary_rows)}",
        "",
    ]
    text_out = "\n".join(header + lines)

    summary_table = (
        "<h2>Summary</h2><table><tr>"
        "<th>spk</th><th>N</th><th>sum_gt</th><th>sum_pred</th><th>ratio</th>"
        "<th>MAE</th><th>Pearson</th><th>Spearman</th><th>cosine</th>"
        "<th>|Δ|≤1</th><th>|Δ|≤2</th><th>norm</th></tr>"
        + "".join(
            f"<tr><td>{esc(s['spk'])}</td><td>{s['n']}</td>"
            f"<td>{s['sum_gt']}</td><td>{s['sum_pred']}</td><td>{s['ratio']:.3f}</td>"
            f"<td>{s['mae']:.3f}</td><td>{s['pearson']:.3f}</td>"
            f"<td>{s['spearman']:.3f}</td><td>{s['cosine']:.3f}</td>"
            f"<td>{s['within1']:.0%}</td><td>{s['within2']:.0%}</td>"
            f"<td class='ph'>{esc(s['norm'])}</td></tr>"
            for s in summary_rows
        )
        + "</table>"
    )

    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Duration sequence similarity</title>",
        "<style>body{font-family:sans-serif;max-width:1100px;margin:24px auto;padding:0 16px}",
        "table{border-collapse:collapse;width:100%;font-size:12px;margin:12px 0}",
        "td,th{border:1px solid #ccc;padding:3px 6px;text-align:right}",
        "th:first-child,td:first-child,td.ph{text-align:left}",
        "code{font-size:11px}.ok{color:green}.warn{color:#a60}",
        ".mono{font-family:Consolas,monospace;letter-spacing:1px}</style></head><body>",
        "<h1>Схожість duration: d_gt vs Kokoro LSTM</h1>",
        "<p><b>sum_gt</b> = сума d_gt після норм (реальний темп LibriTTS). "
        "<b>sum_pred</b> = сума LSTM Kokoro для того ж <code>phonemes_st2</code>. "
        "Pearson/Spearman — схожість <i>форми</i> послідовності (де довгі/короткі фонеми).</p>",
        summary_table,
        *detail_html,
        "</body></html>",
    ]

    out_txt = out_dir / "sequence_compare.txt"
    out_html = out_dir / "sequence_compare.html"
    out_txt.write_text(text_out, encoding="utf-8")
    out_html.write_text("\n".join(html), encoding="utf-8")
    # Console-safe summary (no block sparklines)
    for s in summary_rows:
        print(
            f"spk={s['spk']} N={s['n']} sum_gt={s['sum_gt']} sum_pred={s['sum_pred']} "
            f"ratio={s['ratio']:.3f} MAE={s['mae']:.3f} "
            f"pearson={s['pearson']:.3f} spearman={s['spearman']:.3f} cos={s['cosine']:.3f} "
            f"within1={s['within1']:.0%} within2={s['within2']:.0%}",
            flush=True,
        )
    if ratios and pearsons:
        print(
            f"MEAN ratio pred/gt={float(np.mean(ratios)):.3f} pearson={float(np.mean(pearsons)):.3f}",
            flush=True,
        )
    print(f"Wrote {out_txt}", flush=True)
    print(f"Wrote {out_html}", flush=True)


if __name__ == "__main__":
    main()
