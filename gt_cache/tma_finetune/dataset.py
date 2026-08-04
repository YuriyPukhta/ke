"""Dataset for TMA fine-tune (batch_size=1 friendly; variable N/T)."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from gt_cache.jdc_finetune.common import read_jsonl
from gt_cache.tma_finetune.common import durs_to_hard, rescale_durs_to_t


class TmaKokoroDurDataset(Dataset):
    def __init__(self, data_root: Path | str, split: str = "train", max_mel: int = 0):
        self.data_root = Path(data_root)
        path = self.data_root / f"{split}.jsonl"
        if not path.is_file():
            path = self.data_root / "manifest.jsonl"
        self.rows = read_jsonl(path)
        if not self.rows:
            raise FileNotFoundError(f"empty split: {path}")
        self.max_mel = int(max_mel) if max_mel else 0

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        obj = torch.load(self.data_root / row["path"], map_location="cpu", weights_only=False)
        mel = obj["mel"].float()  # [80, T]
        d_kokoro = obj["d_kokoro"].float().view(-1)
        text_ids = obj["text_ids"].long().view(-1)
        phonemes = obj.get("phonemes", "")

        t = int(mel.shape[-1])
        if self.max_mel > 0 and t > self.max_mel:
            # keep prefix (alignment is left-to-right); trim mel + rescale target later
            mel = mel[:, : self.max_mel]
            t = self.max_mel

        if t % 2 == 1:
            mel = mel[:, :-1]
            t = int(mel.shape[-1])

        n_down = 1
        t_down = max(t // (2**n_down), 1)
        n = min(int(text_ids.numel()), int(d_kokoro.numel()), len(phonemes) or 10**9)
        text_ids = text_ids[:n]
        d_list = [max(1, int(round(float(x)))) for x in d_kokoro[:n].tolist()]
        d_tgt = rescale_durs_to_t(d_list, t_down)
        hard = durs_to_hard(d_tgt).squeeze(0)  # [N, T_down]

        return {
            "mel": mel.contiguous(),
            "text_ids": text_ids.contiguous(),
            "hard": hard.contiguous(),
            "d_kokoro": torch.tensor(d_list, dtype=torch.float32),
            "d_tgt": torch.tensor(d_tgt, dtype=torch.float32),
            "id": row.get("id", str(idx)),
            "voice": row.get("voice", ""),
            "phonemes": phonemes[:n] if phonemes else "",
        }
