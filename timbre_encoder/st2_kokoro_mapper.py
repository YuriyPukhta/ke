"""Map StyleTTS2 acoustic style s_a[128] → Kokoro timbre[:128]."""
from __future__ import annotations

import torch
import torch.nn as nn


class St2ToKokoroMapper(nn.Module):
    """
    Map ST2 s_a → Kokoro timbre[:128].

    residual: timbre = baseline + scale * tanh(MLP(s_a))  (safe, stays near Kokoro)
    absolute: timbre = MLP(s_a)  (can move farther; needs good teachers)
    """

    def __init__(
        self,
        dim: int = 128,
        hidden: int = 512,
        residual_scale: float = 0.35,
        mode: str = "residual",
    ):
        super().__init__()
        self.dim = dim
        self.residual_scale = float(residual_scale)
        self.mode = mode
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        if mode == "residual":
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, s_a_st2: torch.Tensor, baseline_timbre: torch.Tensor | None = None) -> torch.Tensor:
        if s_a_st2.dim() == 1:
            s_a_st2 = s_a_st2.unsqueeze(0)
        out = self.net(s_a_st2)
        if self.mode == "absolute":
            return out
        if baseline_timbre is None:
            raise ValueError("baseline_timbre required for residual mode")
        if baseline_timbre.dim() == 1:
            baseline_timbre = baseline_timbre.unsqueeze(0)
        if baseline_timbre.shape[0] == 1 and s_a_st2.shape[0] > 1:
            baseline_timbre = baseline_timbre.expand(s_a_st2.shape[0], -1)
        return baseline_timbre + self.residual_scale * torch.tanh(out)
