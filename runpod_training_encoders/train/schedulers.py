"""Warmup + cosine learning-rate schedules."""

from __future__ import annotations

import math
from typing import Iterable

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine_lambda(
    num_training_steps: int,
    warmup_ratio: float = 0.05,
    min_ratio: float = 0.1,
):
    warmup_steps = max(1, int(num_training_steps * warmup_ratio))

    def fn(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, num_training_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return fn


def build_warmup_cosine_scheduler(
    optimizer: Optimizer,
    num_training_steps: int,
    warmup_ratio: float = 0.05,
    min_ratio: float = 0.1,
) -> LambdaLR:
    return LambdaLR(
        optimizer,
        lr_lambda=warmup_cosine_lambda(num_training_steps, warmup_ratio, min_ratio),
    )


def build_multi_schedulers(
    optimizers: dict[str, Optimizer],
    num_training_steps: int,
    warmup_ratio: float = 0.05,
    min_ratio: float = 0.1,
) -> dict[str, LambdaLR]:
    return {
        key: build_warmup_cosine_scheduler(opt, num_training_steps, warmup_ratio, min_ratio)
        for key, opt in optimizers.items()
    }


def set_group_lr(optimizer: Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr
        group["initial_lr"] = lr
