"""Build Kokoro/StyleTTS2 model graph from config (imports third_party read-only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from munch import Munch

from runpod_training.paths import ensure_styletts2_on_path
from runpod_training.train.style_init import load_style_shared_from_checkpoint


def recursive_munch(d: Any) -> Any:
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    if isinstance(d, list):
        return [recursive_munch(v) for v in d]
    return d


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_training_model(config: dict, device: torch.device):
    st2 = ensure_styletts2_on_path()
    from models import build_model, load_ASR_models, load_F0_models
    from Utils.PLBERT.util import load_plbert

    def _resolve(p: str) -> str:
        path = Path(p)
        if path.is_file():
            return str(path)
        cand = st2 / p
        if cand.is_file() or cand.is_dir():
            return str(cand)
        return str(path)

    text_aligner = load_ASR_models(_resolve(config["ASR_path"]), _resolve(config["ASR_config"]))
    pitch_extractor = load_F0_models(_resolve(config["F0_path"]))
    plbert = load_plbert(_resolve(config["PLBERT_dir"]))
    model_params = recursive_munch(config["model_params"])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    for key in model:
        model[key].to(device)
    return model, model_params


def load_kokoro_base(model: dict, checkpoint: str | Path) -> None:
    from models import load_checkpoint as st2_load

    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    # load_only_params resets epoch; we only want nets
    st2_load(model, None, str(path), load_only_params=True, ignore_modules=[])


def maybe_init_style_encoder(model: dict, config: dict) -> dict:
    style_cfg = config.get("style_init", {})
    path = style_cfg.get("kikiri_stage1_checkpoint")
    if not path:
        return {"ok": False, "skipped": True}
    return load_style_shared_from_checkpoint(
        model["style_encoder"],
        path,
        reinits_unshared=bool(style_cfg.get("reinit_unshared", True)),
    )
