"""Pure-Python / NumPy MAS (VITS/Glow-TTS) without Cython."""
from __future__ import annotations

import numpy as np
import torch


def mask_from_lens(attn: torch.Tensor, text_lengths: torch.Tensor, mel_lengths: torch.Tensor) -> torch.Tensor:
    """attn [B, T_text, T_mel] -> float mask of valid region."""
    b, tt, tm = attn.shape
    device = attn.device
    t_mask = torch.arange(tt, device=device).unsqueeze(0) < text_lengths.unsqueeze(1)
    m_mask = torch.arange(tm, device=device).unsqueeze(0) < mel_lengths.unsqueeze(1)
    return (t_mask.unsqueeze(-1) & m_mask.unsqueeze(1)).float()


def _maximum_path_each(path: np.ndarray, value: np.ndarray, t_y: int, t_x: int, max_neg_val: float = -1e9) -> None:
    """
    ESPnet/VITS MAS on value[path] shaped [T_mel, T_text].
    Each mel frame (y) is assigned to exactly one text token (x).
    """
    index = t_x - 1
    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            if x == y:
                v_cur = max_neg_val
            else:
                v_cur = value[y - 1, x]
            if x == 0:
                v_prev = 0.0 if y == 0 else max_neg_val
            else:
                v_prev = value[y - 1, x - 1]
            value[y, x] += max(v_prev, v_cur)

    for y in range(t_y - 1, -1, -1):
        path[y, index] = 1
        if index != 0 and (index == y or value[y - 1, index] < value[y - 1, index - 1]):
            index = index - 1


def maximum_path(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Drop-in for StyleTTS2 / VITS maximum_path.
    neg_cent/mask: [B, T_text, T_mel]  (StyleTTS2 layout)
    Returns hard path with exactly one 1 per mel frame (sum == T_mel).
    """
    device = neg_cent.device
    dtype = neg_cent.dtype
    # Work in [B, T_mel, T_text] like ESPnet
    values = neg_cent.detach().float().transpose(1, 2).cpu().numpy().copy()
    masks = mask.detach().float().transpose(1, 2).cpu().numpy()
    paths = np.zeros_like(values, dtype=np.int32)

    # After transpose: [B, mel, text] — same as ESPnet (T_feats, T_text)
    t_ys = masks.sum(axis=1)[:, 0].astype(np.int32)  # mel lengths
    t_xs = masks.sum(axis=2)[:, 0].astype(np.int32)  # text lengths

    for b in range(paths.shape[0]):
        ty, tx = int(t_ys[b]), int(t_xs[b])
        if ty < 1 or tx < 1:
            continue
        _maximum_path_each(paths[b, :ty, :tx], values[b, :ty, :tx], ty, tx)

    out = torch.from_numpy(paths.astype(np.float32)).to(device=device, dtype=dtype)
    return out.transpose(1, 2)  # back to [B, T_text, T_mel]
