"""Compare Kokoro G2P/vocab symbols vs StyleTTS2 TextCleaner (aligner)."""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from timbre_encoder.aligner import ST2_SYMBOL_TO_ID, TextCleaner  # noqa: E402


def main() -> None:
    cfg_path = hf_hub_download("hexgrad/Kokoro-82M", "config.json")
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    kokoro_vocab: dict[str, int] = cfg["vocab"]
    kokoro_syms = set(kokoro_vocab.keys())
    st2_syms = set(ST2_SYMBOL_TO_ID.keys())

    only_kokoro = sorted(kokoro_syms - st2_syms, key=lambda c: (len(c), c))
    only_st2 = sorted(st2_syms - kokoro_syms, key=lambda c: (len(c), c))
    both = kokoro_syms & st2_syms
    multi = [k for k in kokoro_syms if len(k) > 1]

    lines: list[str] = []
    lines.append(f"Kokoro vocab size: {len(kokoro_syms)} (n_token={cfg.get('n_token')})")
    lines.append(f"ST2 TextCleaner size: {len(st2_syms)}")
    lines.append(f"Intersection: {len(both)}")
    lines.append(f"Only in Kokoro (NOT in ST2 aligner): {len(only_kokoro)}")
    lines.append(f"Only in ST2 (NOT in Kokoro vocab): {len(only_st2)}")
    lines.append(f"Kokoro multi-char tokens: {multi}")
    lines.append("")
    lines.append("=== ONLY KOKORO (aligner TextCleaner silently DROPS these) ===")
    for ch in only_kokoro:
        lines.append(f"{ch!r} kokoro_id={kokoro_vocab[ch]}")

    ipa_only_st2 = [
        c
        for c in only_st2
        if (len(c) == 1 and ord(c) > 127)
        or c in set("ˈˌːᵻʰʲʷˠˤ˞̩")
    ]
    lines.append("")
    lines.append(f"=== IPA/special ONLY in ST2 ({len(ipa_only_st2)}) ===")
    lines.append(" ".join(repr(c) for c in ipa_only_st2))

    cleaner = TextCleaner()
    missing_st2: Counter[str] = Counter()
    missing_kokoro: Counter[str] = Counter()
    rows_checked = 0
    len_mismatch = 0
    drop_examples: list[str] = []

    manifests = [
        ROOT / "timbre_encoder/manifests/dev-clean.filtered.csv",
        ROOT / "timbre_encoder/manifests/train-clean-100.filtered.csv",
    ]
    for csv_path in manifests:
        if not csv_path.is_file():
            continue
        with csv_path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                ph = row.get("phonemes") or ""
                if not ph or row.get("align_status") != "ok":
                    continue
                rows_checked += 1
                ids = cleaner(ph)
                dropped = [c for c in ph if c not in ST2_SYMBOL_TO_ID]
                for c in dropped:
                    missing_st2[c] += 1
                if dropped and len(drop_examples) < 20:
                    drop_examples.append(
                        f"{csv_path.name}: dropped={dropped!r} ph[:60]={ph[:60]!r}"
                    )
                if len(ids) != len(ph):
                    len_mismatch += 1
                for c in ph:
                    if c not in kokoro_vocab:
                        missing_kokoro[c] += 1

    pct = 100.0 * len_mismatch / max(1, rows_checked)
    lines.append("")
    lines.append(f"=== CACHED PHONEMES (align_status=ok) rows={rows_checked} ===")
    lines.append(
        f"rows where ST2 len(ids) != len(phonemes): {len_mismatch} ({pct:.2f}%)"
    )
    lines.append(f"chars dropped by ST2: {missing_st2.most_common(40)}")
    lines.append(f"chars missing from Kokoro vocab: {missing_kokoro.most_common(40)}")
    lines.append("drop examples:")
    lines.extend(drop_examples)

    # Spot-check: Kokoro encode vs ST2 encode length on a few rows
    lines.append("")
    lines.append("=== SPOT CHECK first 5 ok rows (dev) ===")
    dev = ROOT / "timbre_encoder/manifests/dev-clean.filtered.csv"
    n = 0
    with dev.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("align_status") != "ok":
                continue
            ph = row["phonemes"]
            st2_ids = cleaner(ph)
            k_ids = [kokoro_vocab.get(c) for c in ph]
            k_ok = [i for i in k_ids if i is not None]
            lines.append(
                f"len_ph={len(ph)} len_st2={len(st2_ids)} len_kokoro={len(k_ok)} "
                f"st2_drop={len(ph) - len(st2_ids)} kokoro_drop={len(ph) - len(k_ok)}"
            )
            n += 1
            if n >= 5:
                break

    out = ROOT / "timbre_encoder/runs/phoneme_vocab_compare.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"WROTE {out}")
    print(f"only_kokoro={len(only_kokoro)} len_mismatch={len_mismatch}/{rows_checked} ({pct:.2f}%)")
    print(f"st2_drops={missing_st2.most_common(10)}")
    print(f"kokoro_miss={missing_kokoro.most_common(10)}")


if __name__ == "__main__":
    main()
