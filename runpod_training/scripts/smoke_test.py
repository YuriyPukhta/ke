#!/usr/bin/env python3
"""Smoke tests: index QC sample, 2 train steps, checkpoint round-trip."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch
import yaml

from runpod_training.data.npz_dataset import NPZTargetsDataset, build_npz_dataloader
from runpod_training.data.prepare_index import audit_npz
from runpod_training.paths import ensure_styletts2_on_path, ensure_volume
from runpod_training.train.checkpointing import load_checkpoint, save_checkpoint
from runpod_training.train.freeze import apply_phase, trainable_param_count
from runpod_training.train.model_factory import build_training_model, load_kokoro_base, load_yaml


def smoke_index(index_jsonl: Path, limit: int = 8) -> dict:
    rows = [json.loads(l) for l in index_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    ok = 0
    for row in rows[:limit]:
        audit_npz(Path(row["npz_path"]))
        ok += 1
    return {"checked": ok, "total": len(rows)}


def smoke_dataloader(index_jsonl: Path, data_root: Path) -> dict:
    loader = build_npz_dataloader(
        index_jsonl, data_root, batch_size=2, num_workers=0, validation=True, speaker_balanced=False
    )
    batch = next(iter(loader))
    texts, mels = batch[1], batch[5]
    assert texts.dtype == torch.long
    assert mels.dim() == 3
    return {"batch_texts": list(texts.shape), "batch_mels": list(mels.shape)}


def smoke_train_steps(config: dict, steps: int = 2) -> dict:
    ensure_styletts2_on_path()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = build_training_model(config, device)
    if Path(config["pretrained_model"]).is_file():
        load_kokoro_base(model, config["pretrained_model"])
    apply_phase(model, "aligner_warmup")
    loader = build_npz_dataloader(
        config["data_params"]["train_index"],
        config["data_params"]["root_path"],
        batch_size=2,
        num_workers=0,
        validation=True,
        speaker_balanced=False,
    )
    n_down = model["text_aligner"].n_down
    from runpod_training.train.train_stage1 import length_to_mask

    losses = []
    it = iter(loader)
    for _ in range(steps):
        batch = next(it)
        texts = batch[1].to(device)
        input_lengths = batch[2].to(device)
        mels = batch[5].to(device)
        mel_input_length = batch[6].to(device)
        mask = length_to_mask(mel_input_length // (2**n_down)).to(device)
        ppgs, s2s_pred, s2s_attn = model["text_aligner"](mels, mask, texts)
        loss = s2s_pred.float().pow(2).mean() + s2s_attn.float().pow(2).mean()
        assert torch.isfinite(loss)
        loss.backward()
        losses.append(float(loss.detach().cpu()))
    # checkpoint round-trip
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pth"
        save_checkpoint(path, model, None, None, 1, steps, "smoke")
        load_checkpoint(path, model, load_optimizer=False)
    return {"losses": losses, "trainable": trainable_param_count(model)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    ensure_volume(Path(cfg.get("run_root", "/workspace/kokoro-run")))
    report = {
        "index": smoke_index(Path(cfg["data_params"]["train_index"])),
        "loader": smoke_dataloader(
            Path(cfg["data_params"]["train_index"]), Path(cfg["data_params"]["root_path"])
        ),
    }
    if not args.skip_train:
        report["train_steps"] = smoke_train_steps(cfg, 2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
