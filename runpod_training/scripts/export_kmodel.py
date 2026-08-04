#!/usr/bin/env python3
"""Export StyleTTS2 training checkpoint nets to Kokoro KModel-compatible state dict."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


KOKORO_KEYS = ("bert", "bert_encoder", "predictor", "decoder", "text_encoder")


def export(checkpoint: Path, output: Path, config_json: Path | None = None) -> Path:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    net = state.get("net", state)
    out = {}
    for key in KOKORO_KEYS:
        if key not in net:
            raise KeyError(f"missing {key} in checkpoint")
        # KModel often expects module. prefix historically; keep clean keys
        out[key] = {k.removeprefix("module."): v.cpu() for k, v in net[key].items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output)
    if config_json and config_json.is_file():
        (output.parent / "config.json").write_bytes(config_json.read_bytes())
    meta = {
        "source": str(checkpoint),
        "keys": list(out),
        "epoch": state.get("epoch"),
        "iters": state.get("iters"),
    }
    (output.parent / "export_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"exported {output}")
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config-json", type=Path, default=None)
    args = p.parse_args()
    export(args.checkpoint, args.output, args.config_json)


if __name__ == "__main__":
    main()
