"""Initialize style_encoder.shared from StyleTTS2 multi-speaker Stage 1 checkpoint."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _strip_module(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k.removeprefix("module."): v for k, v in state.items()}


def _convert_spectral_norm_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map old weight_norm/spectral_norm keys to parametrizations API when possible."""
    out = {}
    for key, value in state.items():
        new_key = key
        # Common legacy patterns → parametrizations
        if ".weight_orig" in key:
            new_key = key.replace(".weight_orig", ".parametrizations.weight.original0")
        elif key.endswith(".weight_u"):
            # drop u/v for spectral_norm; parametrizations recompute
            continue
        elif key.endswith(".weight_v"):
            continue
        out[new_key] = value
    return out


def load_style_shared_from_checkpoint(
    style_encoder: nn.Module,
    checkpoint_path: str | Path,
    reinits_unshared: bool = True,
) -> dict[str, Any]:
    """Load only style_encoder.shared; re-init unshared projection.

    Returns audit dict. On failure, leaves encoder unchanged and sets ok=False.
    """
    path = Path(checkpoint_path)
    audit: dict[str, Any] = {"path": str(path), "ok": False, "loaded_keys": 0, "missing": [], "unexpected": []}
    if not path.is_file():
        audit["error"] = f"missing checkpoint: {path}"
        logger.warning("%s", audit["error"])
        return audit

    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
        net = raw.get("net", raw)
        if "style_encoder" not in net:
            audit["error"] = "checkpoint has no style_encoder"
            logger.warning("%s", audit["error"])
            return audit
        state = _strip_module(net["style_encoder"])
        state = _convert_spectral_norm_keys(state)
        shared_state = {k: v for k, v in state.items() if k.startswith("shared.")}
        if not shared_state:
            # maybe already without prefix nesting
            shared_state = {f"shared.{k}": v for k, v in state.items() if not k.startswith("unshared")}

        current = style_encoder.state_dict()
        filtered = {}
        for k, v in shared_state.items():
            if k in current and current[k].shape == v.shape:
                filtered[k] = v
            else:
                audit["missing"].append(k)

        missing, unexpected = style_encoder.load_state_dict(filtered, strict=False)
        audit["loaded_keys"] = len(filtered)
        audit["missing"] = list(missing)
        audit["unexpected"] = list(unexpected)

        if reinits_unshared and hasattr(style_encoder, "unshared"):
            if hasattr(style_encoder.unshared, "reset_parameters"):
                style_encoder.unshared.reset_parameters()
            else:
                for p in style_encoder.unshared.parameters():
                    if p.dim() >= 2:
                        nn.init.xavier_uniform_(p)
                    else:
                        nn.init.zeros_(p)

        # Smoke: forward random mel-like tensor
        with torch.no_grad():
            x = torch.randn(1, 1, 80, 192)
            y = style_encoder(x)
            if not torch.isfinite(y).all():
                raise RuntimeError("style_encoder forward produced non-finite values")
            audit["out_norm"] = float(y.norm().item())
        audit["ok"] = audit["loaded_keys"] > 0
        logger.info(
            "style_encoder.shared transfer: ok=%s keys=%d out_norm=%.4f",
            audit["ok"],
            audit["loaded_keys"],
            audit.get("out_norm", -1),
        )
    except Exception as exc:
        audit["error"] = str(exc)
        audit["ok"] = False
        logger.warning("style_encoder transfer failed, keeping random init: %s", exc)
    return audit
