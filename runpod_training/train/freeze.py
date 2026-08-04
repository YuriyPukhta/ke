"""Freeze / unfreeze policies for Stage 1 warm-up and Stage 2."""

from __future__ import annotations

from typing import Iterable

import torch.nn as nn


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def freeze_all(model: dict) -> None:
    for key in model:
        set_requires_grad(model[key], False)


def unfreeze(model: dict, keys: Iterable[str]) -> None:
    for key in keys:
        if key in model:
            set_requires_grad(model[key], True)


def apply_phase(model: dict, phase: str) -> None:
    """phase: aligner_warmup | stage1 | stage2_pre_joint | stage2_joint"""
    freeze_all(model)
    if phase == "aligner_warmup":
        unfreeze(model, ["text_aligner"])
    elif phase == "stage1":
        unfreeze(model, ["text_aligner", "text_encoder", "style_encoder", "decoder", "mpd", "msd"])
        # JDC stays frozen (teacher)
        set_requires_grad(model["pitch_extractor"], False)
        # PLBERT / predictor unused in stage1
        set_requires_grad(model["bert"], False)
        set_requires_grad(model["bert_encoder"], False)
        set_requires_grad(model["predictor"], False)
        set_requires_grad(model["predictor_encoder"], False)
    elif phase == "stage2_pre_joint":
        unfreeze(model, ["bert_encoder", "predictor", "predictor_encoder"])
        set_requires_grad(model["bert"], False)
        set_requires_grad(model["text_encoder"], False)
        set_requires_grad(model["text_aligner"], False)
        set_requires_grad(model["pitch_extractor"], False)
        set_requires_grad(model["style_encoder"], False)
        set_requires_grad(model["decoder"], False)
    elif phase == "stage2_joint":
        unfreeze(
            model,
            [
                "bert_encoder",
                "predictor",
                "predictor_encoder",
                "style_encoder",
                "decoder",
                "mpd",
                "msd",
                "wd",
            ],
        )
        set_requires_grad(model["bert"], False)
        set_requires_grad(model["text_encoder"], False)
        set_requires_grad(model["text_aligner"], False)
        set_requires_grad(model["pitch_extractor"], False)
    else:
        raise ValueError(f"unknown phase: {phase}")


def trainable_param_count(model: dict) -> int:
    total = 0
    for key in model:
        total += sum(p.numel() for p in model[key].parameters() if p.requires_grad)
    return total
