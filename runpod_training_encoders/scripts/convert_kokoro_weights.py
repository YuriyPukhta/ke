#!/usr/bin/env python3
"""Convert hexgrad/Kokoro-82M weights to StyleTTS2 {'net': ...} checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download


def convert(output: Path, force: bool = False) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        print(f"exists: {output}")
        return output
    model_path = hf_hub_download("hexgrad/Kokoro-82M", "kokoro-v1_0.pth")
    config_path = hf_hub_download("hexgrad/Kokoro-82M", "config.json")
    raw = torch.load(model_path, map_location="cpu", weights_only=True)
    net = {}
    for component, state in raw.items():
        net[component] = {k.removeprefix("module."): v for k, v in state.items()}
        print(f"  {component}: {len(net[component])} tensors")
    torch.save({"net": net}, output)
    cfg_out = output.with_name("config.json")
    cfg_out.write_bytes(Path(config_path).read_bytes())
    print(f"saved {output}")
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("/workspace/kokoro-run/weights/kokoro_base.pth"))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    convert(args.output, args.force)


if __name__ == "__main__":
    main()
