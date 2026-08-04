"""Dataset / collate for JDC fine-tune."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler

from gt_cache.jdc_finetune.common import read_jsonl


class JdcKokoroF0Dataset(Dataset):
    def __init__(
        self,
        data_root: Path | str,
        split: str = "train",
        max_frames: int = 0,
    ):
        self.data_root = Path(data_root)
        path = self.data_root / f"{split}.jsonl"
        if not path.is_file():
            path = self.data_root / "manifest.jsonl"
        self.rows = read_jsonl(path)
        if not self.rows:
            raise FileNotFoundError(f"empty split: {path}")
        self.max_frames = int(max_frames) if max_frames else 0

    def __len__(self) -> int:
        return len(self.rows)

    def lengths(self) -> list[int]:
        out = []
        for r in self.rows:
            t = r.get("T")
            out.append(int(t) if t not in (None, "") else 0)
        return out

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        obj = torch.load(self.data_root / row["path"], map_location="cpu", weights_only=False)
        mel = obj["mel"].float()  # [80, T]
        f0 = obj["f0"].float().view(-1)
        t = min(mel.shape[-1], f0.numel())
        if self.max_frames > 0 and t > self.max_frames:
            # deterministic center crop keeps pitch contour mid-utterance
            start = (t - self.max_frames) // 2
            end = start + self.max_frames
            mel = mel[:, start:end]
            f0 = f0[start:end]
            t = self.max_frames
        return {
            "mel": mel[:, :t].contiguous(),
            "f0": f0[:t].contiguous(),
            "id": row.get("id", str(idx)),
            "voice": row.get("voice", ""),
        }


def collate_pad(batch: list[dict]) -> dict:
    """Pad variable-length mels along time; return mask [B, T] True=valid."""
    lengths = [int(b["mel"].shape[-1]) for b in batch]
    t_max = max(lengths)
    bsz = len(batch)
    mel = torch.zeros(bsz, 80, t_max, dtype=torch.float32)
    f0 = torch.zeros(bsz, t_max, dtype=torch.float32)
    mask = torch.zeros(bsz, t_max, dtype=torch.bool)
    for i, item in enumerate(batch):
        t = lengths[i]
        mel[i, :, :t] = item["mel"]
        f0[i, :t] = item["f0"]
        mask[i, :t] = True
    return {
        "mel": mel,
        "f0": f0,
        "mask": mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "ids": [b["id"] for b in batch],
    }


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Group similar-length clips to cut padding / VRAM (shuffle buckets each epoch)."""

    def __init__(self, lengths: list[int], batch_size: int, seed: int = 0):
        self.lengths = lengths
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch)
        order = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i] or 10**9)
        # shuffle within windows of ~4 batches for some randomness
        window = max(self.batch_size * 4, self.batch_size)
        shuffled: list[int] = []
        for start in range(0, len(order), window):
            chunk = order[start : start + window]
            perm = torch.randperm(len(chunk), generator=rng).tolist()
            shuffled.extend(chunk[j] for j in perm)
        for i in range(0, len(shuffled), self.batch_size):
            batch = shuffled[i : i + self.batch_size]
            if batch:
                yield batch
        self.epoch += 1

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size
