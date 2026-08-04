#!/usr/bin/env python3
"""Extract deterministic voicepacks [510,1,256] from trained encoders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from runpod_training.paths import ensure_styletts2_on_path
from runpod_training.train.checkpointing import load_checkpoint
from runpod_training.train.model_factory import build_training_model, load_yaml

to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)


def preprocess(wave: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(np.asarray(wave, dtype=np.float32))
    mel = to_mel(x)
    return (torch.log(1e-5 + mel.unsqueeze(0)) - (-4)) / 4


def load_rows(index: Path) -> list[dict]:
    return [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line.strip()]


@torch.no_grad()
def extract_for_speaker(model, rows, data_root: Path, speaker: int, n_samples: int, device) -> torch.Tensor:
    pool = [r for r in rows if int(r["speaker_id"]) == speaker]
    pool = sorted(pool, key=lambda r: r["audio"])[:n_samples]
    if not pool:
        raise ValueError(f"no rows for speaker {speaker}")
    acoustic, prosodic = [], []
    for row in pool:
        path = Path(row.get("audio_path") or (data_root / row["audio"]))
        wave, sr = sf.read(path.as_posix())
        if getattr(wave, "ndim", 1) > 1:
            wave = wave[:, 0]
        if sr != 24000:
            import librosa

            wave = librosa.resample(wave, orig_sr=sr, target_sr=24000)
        mel = preprocess(wave).to(device)
        acoustic.append(model["style_encoder"](mel.unsqueeze(0)))
        prosodic.append(model["predictor_encoder"](mel.unsqueeze(0)))
    a = torch.stack(acoustic).mean(0)
    p = torch.stack(prosodic).mean(0)
    style = torch.cat([a, p], dim=-1)  # [1, 256]
    # Broadcast to [510, 1, 256]
    return style.view(1, 1, 256).expand(510, 1, 256).contiguous().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--style-checkpoint", type=Path, default=None, help="optional Stage1 for style_encoder")
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--speakers", type=str, default="103")
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args()

    ensure_styletts2_on_path()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_path = args.config
    if cfg_path is None:
        # minimal config skeleton from package
        cfg_path = Path(__file__).resolve().parents[1] / "configs" / "config_libritts_a40.yml"
    config = load_yaml(cfg_path)
    model, _ = build_training_model(config, device)
    load_checkpoint(args.checkpoint, model, load_optimizer=False)
    if args.style_checkpoint and args.style_checkpoint.is_file():
        st = torch.load(args.style_checkpoint, map_location="cpu", weights_only=False)
        net = st.get("net", st)
        if "style_encoder" in net:
            model["style_encoder"].load_state_dict(net["style_encoder"], strict=False)
            print("loaded style_encoder from", args.style_checkpoint)

    for key in model:
        model[key].eval()

    rows = load_rows(args.index)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for spk in [int(x) for x in args.speakers.split(",") if x.strip()]:
        pack = extract_for_speaker(model, rows, args.data_root, spk, args.n_samples, device)
        out = args.output_dir / f"spk_{spk}.pt"
        torch.save(pack, out)
        print(f"wrote {out} shape={tuple(pack.shape)} norm={pack.norm():.3f}")


if __name__ == "__main__":
    main()
