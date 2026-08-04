"""StyleTTS2/StyleTTS2 SLM-adv wiring helpers (diffusion optional)."""

from __future__ import annotations

import torch


class EchoStyleSampler:
    """Stand-in for the diffusion sampler when ``diff_epoch`` is disabled.

    StyleTTS2's ``SLMAdversarialLoss`` still calls ``sampler(...)`` on the
    non-``s_trg`` branch. We echo the ground-truth style (or ``features``)
    so the rest of the StyleTTS2 module stays unchanged.
    """

    def __init__(self) -> None:
        self.style: torch.Tensor | None = None

    def __call__(
        self,
        noise,
        embedding,
        embedding_scale=1,
        features=None,
        embedding_mask_proba=0.1,
        num_steps=3,
    ):
        if features is not None:
            return features.unsqueeze(1) if features.dim() == 2 else features
        if self.style is None:
            raise RuntimeError("EchoStyleSampler.style was not set before SLM-adv call")
        s = self.style
        return s.unsqueeze(1) if s.dim() == 2 else s
