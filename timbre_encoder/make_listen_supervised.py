"""Listen test for supervised Kokoro-space encoder (should sound like speech)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo"))

from kokoro.pipeline import KPipeline


def synth(pipeline, text, voice, path: Path):
    chunks = []
    for r in pipeline(text, voice=voice, speed=1):
        if r.audio is not None:
            chunks.append(r.audio.cpu().numpy())
    audio = np.concatenate(chunks)
    sf.write(str(path), audio, 24000)
    return path


def main() -> None:
    out = ROOT / "timbre_encoder" / "runs" / "supervised_kokoro" / "listen_test"
    out.mkdir(parents=True, exist_ok=True)
    ckpt = ROOT / "timbre_encoder" / "runs" / "supervised_kokoro" / "dual_encoder_final.pt"
    pipeline = KPipeline(lang_code="a")

    cases = [
        ("am_michael", "The stars direct these actions, as in other affairs."),
        ("af_bella", "He hurriedly cut cake and pressed it upon her."),
        ("bm_george", "After he met Winifred Pemberton he seemed to himself like a different man."),
    ]

    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Supervised encoder listen</title></head><body>",
        "<h1>Supervised Kokoro-space encoder</h1>",
        "<p>Teacher voice synth → encode mel → resynth with predicted timbre + af_heart prosody.</p>",
    ]

    for i, (voice, text) in enumerate(cases, 1):
        teacher_wav = out / f"{i:02d}_{voice}_teacher.wav"
        encoded_wav = out / f"{i:02d}_{voice}_encoded.wav"
        heart_wav = out / f"{i:02d}_{voice}_af_heart_same_text.wav"
        print(f"=== {voice} ===", flush=True)
        synth(pipeline, text, voice, teacher_wav)
        synth(pipeline, text, "af_heart", heart_wav)
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "timbre_encoder.infer_dual",
                "--wav",
                str(teacher_wav),
                "--text",
                text,
                "--ckpt",
                str(ckpt),
                "--out",
                str(encoded_wav),
                "--mode",
                "timbre_only",
                "--baseline",
                "af_heart",
            ],
            cwd=str(ROOT),
        )
        html += [
            f"<h3>#{i} {voice}</h3><p>{text}</p>",
            "<p><b>Teacher (official Kokoro voice)</b></p>",
            f"<audio controls src='{teacher_wav.name}'></audio>",
            "<p><b>Encoded from teacher wav</b></p>",
            f"<audio controls src='{encoded_wav.name}'></audio>",
            "<p><b>Same text with af_heart (control)</b></p>",
            f"<audio controls src='{heart_wav.name}'></audio><hr>",
        ]

    html.append("</body></html>")
    (out / "index.html").write_text("\n".join(html), encoding="utf-8")
    print(f"DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
