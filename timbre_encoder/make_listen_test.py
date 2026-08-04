"""Generate original vs encoder-synthesized pairs for listening on filtered dev."""
from __future__ import annotations

import csv
import random
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest = ROOT / "timbre_encoder" / "manifests" / "dev-clean.filtered.csv"
    ckpt = ROOT / "timbre_encoder" / "runs" / "dual_snr22_residual" / "dual_encoder_final.pt"
    out_dir = ROOT / "timbre_encoder" / "runs" / "dual_snr22_residual" / "listen_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    cands = [r for r in rows if 40 <= len(r["text"]) <= 120]
    random.seed(42)
    samples = random.sample(cands, min(5, len(cands)))

    index: list[dict] = []
    for i, r in enumerate(samples, 1):
        wav = Path(r["audio_path"])
        text = r["text"]
        spk = r["speaker_id"]
        snr = r.get("snr", "")
        orig = out_dir / f"{i:02d}_spk{spk}_original.wav"
        synth = out_dir / f"{i:02d}_spk{spk}_encoded.wav"
        shutil.copy2(wav, orig)
        print(f"=== [{i}] spk={spk} snr={snr} ===", flush=True)
        print(f"text: {text}", flush=True)
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "timbre_encoder.infer_dual",
                "--wav",
                str(wav),
                "--text",
                text,
                "--ckpt",
                str(ckpt),
                "--out",
                str(synth),
                "--mode",
                "timbre_only",
                "--baseline",
                "af_heart",
            ],
            cwd=str(ROOT),
        )
        index.append(
            {
                "i": i,
                "spk": spk,
                "snr": snr,
                "text": text,
                "original": orig.name,
                "encoded": synth.name,
            }
        )

    lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Dual encoder listen test</title></head><body>",
        "<h1>Dual encoder listen test</h1>",
        "<p>Original LibriTTS-R vs Kokoro DualStyleEncoder residual (timbre_only + af_heart).</p>",
    ]
    for it in index:
        lines.append(f"<h3>#{it['i']} speaker {it['spk']} (SNR {it['snr']})</h3>")
        lines.append(f"<p>{it['text']}</p>")
        lines.append("<p><b>Original</b></p>")
        lines.append(f"<audio controls src='{it['original']}'></audio>")
        lines.append("<p><b>Encoded</b></p>")
        lines.append(f"<audio controls src='{it['encoded']}'></audio><hr>")
    lines.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(lines), encoding="utf-8")

    readme = ["# Listen test\n", "Open `index.html` in a browser, or play wav pairs:\n"]
    for it in index:
        readme.append(f"## #{it['i']} spk {it['spk']} SNR {it['snr']}\n")
        readme.append(f"- text: {it['text']}\n")
        readme.append(f"- original: `{it['original']}`\n")
        readme.append(f"- encoded: `{it['encoded']}`\n")
    (out_dir / "README.md").write_text("".join(readme), encoding="utf-8")
    print(f"DONE -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
