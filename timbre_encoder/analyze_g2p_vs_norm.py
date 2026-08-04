"""
Compare Kokoro G2P phonemes vs normalized phonemes_st2 in cached manifests.

Usage:
  .venv\\Scripts\\python.exe -m timbre_encoder.analyze_g2p_vs_norm
"""
from __future__ import annotations

import argparse
import csv
import html as html_lib
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from timbre_encoder.phoneme_norm import normalize_kokoro_phonemes_for_st2  # noqa: E402


def analyze_csv(path: Path) -> dict:
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    usable = [
        r
        for r in rows
        if (r.get("phonemes") or "").strip()
        and (r.get("phonemes_st2") or "").strip()
        and r.get("norm_applied") in ("0", "1")
    ]

    identical = 0
    changed = 0
    delta_hist = Counter()
    ops_hist = Counter()  # repl / drop / exp counts aggregated
    char_map_hits = Counter()  # which raw chars got remapped/dropped
    examples_changed = []
    examples_identical = []
    len_raw_sum = 0
    len_st2_sum = 0
    # Recompute norm from raw to verify cache consistency
    cache_mismatch = 0

    for r in usable:
        raw = r["phonemes"]
        st2 = r["phonemes_st2"]
        flag = r["norm_applied"]
        delta = int(r.get("delta_len") or 0)
        ops = r.get("norm_ops") or ""

        nr = normalize_kokoro_phonemes_for_st2(raw)
        if nr.normalized != st2 or (nr.changed and flag != "1") or ((not nr.changed) and flag != "0"):
            cache_mismatch += 1

        len_raw_sum += len(raw)
        len_st2_sum += len(st2)
        delta_hist[delta] += 1

        if raw == st2:
            identical += 1
            if len(examples_identical) < 5:
                examples_identical.append((r.get("speaker_id"), r.get("text", "")[:60], raw))
        else:
            changed += 1
            # parse ops like repl=1|drop=2|dlen=-1
            for part in ops.split("|"):
                if part.startswith("repl="):
                    ops_hist["repl"] += int(part.split("=")[1])
                elif part.startswith("drop="):
                    ops_hist["drop"] += int(part.split("=")[1])
                elif part.startswith("exp="):
                    ops_hist["exp"] += int(part.split("=")[1])

            # which chars differ: walk with NormResult ops details
            for op in nr.ops:
                if op.startswith("map:"):
                    # map:ᵊ->ə
                    left = op[4:].split("->", 1)[0]
                    char_map_hits[f"map:{left}"] += 1
                elif op.startswith("drop:"):
                    char_map_hits[op] += 1
                elif op.startswith("expand:"):
                    left = op.split(":", 1)[1].split("->", 1)[0]
                    char_map_hits[f"expand:{left}"] += 1
                elif op.startswith("drop_unknown:"):
                    char_map_hits[op] += 1

            if len(examples_changed) < 12:
                examples_changed.append(
                    {
                        "spk": r.get("speaker_id"),
                        "text": (r.get("text") or "")[:70],
                        "raw": raw,
                        "st2": st2,
                        "ops": ops,
                        "delta": delta,
                        "len_raw": len(raw),
                        "len_st2": len(st2),
                    }
                )

    n = len(usable)
    return {
        "path": str(path),
        "total_rows": len(rows),
        "usable": n,
        "identical": identical,
        "changed": changed,
        "identical_pct": 100.0 * identical / n if n else 0.0,
        "changed_pct": 100.0 * changed / n if n else 0.0,
        "delta_hist": dict(sorted(delta_hist.items())),
        "ops_hist": dict(ops_hist),
        "char_map_hits": char_map_hits.most_common(20),
        "avg_len_raw": len_raw_sum / n if n else 0.0,
        "avg_len_st2": len_st2_sum / n if n else 0.0,
        "cache_mismatch": cache_mismatch,
        "examples_changed": examples_changed,
        "examples_identical": examples_identical,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csvs",
        nargs="+",
        default=[
            str(ROOT / "timbre_encoder/manifests/dev-clean.filtered.csv"),
            str(ROOT / "timbre_encoder/manifests/train-clean-100.filtered.csv"),
        ],
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "timbre_encoder/runs/g2p_vs_norm"),
    )
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    esc = html_lib.escape
    reports = []
    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Kokoro G2P vs phonemes_st2</title>",
        "<style>body{font-family:sans-serif;max-width:1000px;margin:24px auto;padding:0 16px}",
        "table{border-collapse:collapse;width:100%;font-size:13px} td,th{border:1px solid #ccc;padding:4px 8px;text-align:left}",
        "code{font-size:12px;word-break:break-all}</style></head><body>",
        "<h1>Text → Kokoro G2P → phonemes vs phonemes_st2</h1>",
        "<p>Порівняння сирого Misaki/Kokoro виводу з нормалізованим рядком для ST2 aligner.</p>",
    ]

    for csv_path in args.csvs:
        p = Path(csv_path)
        if not p.is_file():
            print(f"skip missing {p}", flush=True)
            continue
        print(f"analyzing {p} ...", flush=True)
        r = analyze_csv(p)
        reports.append(r)

        lines = [
            f"=== {p.name} ===",
            f"usable={r['usable']}/{r['total_rows']} (have phonemes + phonemes_st2 + norm_applied)",
            f"identical raw==st2: {r['identical']} ({r['identical_pct']:.1f}%)",
            f"changed (norm_applied=1): {r['changed']} ({r['changed_pct']:.1f}%)",
            f"avg len raw={r['avg_len_raw']:.1f} st2={r['avg_len_st2']:.1f}",
            f"delta_len histogram: {r['delta_hist']}",
            f"op counts (sum over changed rows): {r['ops_hist']}",
            f"top char ops: {r['char_map_hits']}",
            f"cache_vs_recompute_mismatch: {r['cache_mismatch']}",
            "",
        ]
        text_block = "\n".join(lines)
        (out_dir / f"{p.stem}_g2p_vs_norm.txt").write_text(text_block, encoding="utf-8")
        top_safe = [(k.encode("unicode_escape").decode(), v) for k, v in r["char_map_hits"][:5]]
        print(
            f"{p.name}: usable={r['usable']} identical={r['identical_pct']:.1f}% "
            f"changed={r['changed_pct']:.1f}% avg_len {r['avg_len_raw']:.1f}->{r['avg_len_st2']:.1f} "
            f"delta_hist={r['delta_hist']} ops={r['ops_hist']} top={top_safe} "
            f"cache_mismatch={r['cache_mismatch']}",
            flush=True,
        )

        html += [
            f"<h2>{esc(p.name)}</h2>",
            "<table>",
            f"<tr><th>usable</th><td>{r['usable']}/{r['total_rows']}</td></tr>",
            f"<tr><th>identical (raw==st2)</th><td><b>{r['identical_pct']:.1f}%</b> ({r['identical']})</td></tr>",
            f"<tr><th>changed</th><td><b>{r['changed_pct']:.1f}%</b> ({r['changed']})</td></tr>",
            f"<tr><th>avg len</th><td>raw={r['avg_len_raw']:.1f} → st2={r['avg_len_st2']:.1f}</td></tr>",
            f"<tr><th>delta_len hist</th><td><code>{esc(str(r['delta_hist']))}</code></td></tr>",
            f"<tr><th>ops</th><td><code>{esc(str(r['ops_hist']))}</code></td></tr>",
            f"<tr><th>top char changes</th><td><code>{esc(str(r['char_map_hits']))}</code></td></tr>",
            f"<tr><th>cache mismatch</th><td>{r['cache_mismatch']}</td></tr>",
            "</table>",
            "<h3>Examples: changed</h3>",
        ]
        for ex in r["examples_changed"][:8]:
            html += [
                f"<p><b>spk {esc(str(ex['spk']))}</b> {esc(ex['text'])}<br>"
                f"ops=<code>{esc(ex['ops'])}</code> delta={ex['delta']}<br>"
                f"raw ({ex['len_raw']}): <code>{esc(ex['raw'])}</code><br>"
                f"st2 ({ex['len_st2']}): <code>{esc(ex['st2'])}</code></p>"
            ]
        html.append("<h3>Examples: identical</h3>")
        for spk, text, raw in r["examples_identical"][:3]:
            html.append(
                f"<p><b>spk {esc(str(spk))}</b> {esc(text)}<br>"
                f"<code>{esc(raw)}</code></p>"
            )

    html.append("</body></html>")
    index = out_dir / "index.html"
    index.write_text("\n".join(html), encoding="utf-8")

    # combined short summary
    summary_lines = ["Kokoro G2P vs phonemes_st2", ""]
    for r in reports:
        name = Path(r["path"]).name
        top_safe = [(k.encode("unicode_escape").decode(), v) for k, v in r["char_map_hits"][:3]]
        summary_lines.append(
            f"{name}: identical={r['identical_pct']:.1f}% changed={r['changed_pct']:.1f}% "
            f"avg_len {r['avg_len_raw']:.1f}->{r['avg_len_st2']:.1f} ops={r['ops_hist']} "
            f"top={top_safe}"
        )
    (out_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print("Wrote", index, flush=True)
    print("\n".join(summary_lines), flush=True)


if __name__ == "__main__":
    main()
