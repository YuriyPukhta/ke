"""Check holdout #1 phonemes vs Kokoro vocab."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))
sys.path.insert(0, str(ROOT / "speaker_ref_opt"))

import torch
from kokoro_revers.model import KModel

from optimize_speaker_ref_s import build_alignment, kokoro_ids, load_speaker_rows
from timbre_encoder.librispeech_dataset import parse_d_gt

OUT = Path(__file__).resolve().parent / "runs" / "_phoneme_check_01.txt"


def main() -> None:
    manifest = ROOT / "timbre_encoder/manifests/train-clean-100.filtered.csv"
    rows = load_speaker_rows(manifest, "6209", max_phonemes=100, max_clips=67, seed=0)
    row = rows[1]  # listen #1
    ph = (row.get("phonemes_st2") or "").strip()
    d_gt = parse_d_gt(row.get("d_gt"))
    assert d_gt is not None

    model = KModel(repo_id="hexgrad/Kokoro-82M")
    input_ids, ids, dropped = kokoro_ids(model, ph, torch.device("cpu"))
    aln = build_alignment(d_gt, torch.device("cpu"))

    inv = {v: k for k, v in model.vocab.items() if v is not None}
    rev = "".join(inv.get(i, "?") for i in ids)

    lines: list[str] = [
        f"audio={row['audio_path']}",
        f"text={row.get('text')}",
        f"phonemes_st2_len={len(ph)}",
        f"d_gt_len={len(d_gt)}",
        f"phonemes_st2={ph}",
        f"dropped={dropped}",
        f"n_ids_inner={len(ids)}",
        f"input_ids_len={input_ids.shape[-1]}",
        f"input_ids={input_ids.tolist()}",
        f"aln_shape={tuple(aln.shape)}",
        f"aln_tokens_match_input_ids={aln.shape[1] == input_ids.shape[1]}",
        f"d_gt_sum_core={sum(d_gt)}",
        f"aln_frames={aln.shape[-1]}",
        f"roundtrip_from_ids={rev}",
        f"roundtrip_equals_st2={rev == ph}",
        f"passed_to_kokoro_ok={len(dropped) == 0 and len(ids) == len(ph)}",
        "",
        "i char ord vocab_id d_gt status",
    ]
    miss: list[int] = []
    for i, c in enumerate(ph):
        v = model.vocab.get(c)
        ok = v is not None
        if not ok:
            miss.append(i)
        status = "OK" if ok else "MISS"
        lines.append(f"{i:02d} {c!r} {ord(c)} {v} {d_gt[i]} {status}")
    lines.append(f"missing_count={len(miss)} indices={miss}")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"passed_to_kokoro_ok={len(dropped) == 0 and len(ids) == len(ph)}")
    print(f"aln_ok={aln.shape[1] == input_ids.shape[1]}")
    print(f"dropped={dropped}")
    print(f"len_ph={len(ph)} n_ids={len(ids)} input_ids={input_ids.shape[-1]}")


if __name__ == "__main__":
    main()
