"""Speaker-balanced sampling helpers."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Sequence


def speaker_balanced_indices(
    speakers: Sequence[int],
    epoch: int,
    seed: int = 42,
) -> list[int]:
    """Return shuffled indices with approximately equal speaker exposure.

    Builds one virtual epoch by round-robin over speakers after per-speaker
    shuffle. Shorter speakers cycle so every speaker appears ~equal times.
    """
    by_spk: dict[int, list[int]] = defaultdict(list)
    for idx, spk in enumerate(speakers):
        by_spk[int(spk)].append(idx)
    if not by_spk:
        return []

    rng = random.Random(seed + epoch * 9973)
    pools = {}
    max_count = 0
    for spk, idxs in by_spk.items():
        order = idxs[:]
        rng.shuffle(order)
        pools[spk] = order
        max_count = max(max_count, len(order))

    out: list[int] = []
    speakers_order = sorted(pools)
    rng.shuffle(speakers_order)
    for i in range(max_count):
        for spk in speakers_order:
            pool = pools[spk]
            out.append(pool[i % len(pool)])
    rng.shuffle(out)
    return out


class SpeakerBalancedBatchSampler:
    """Yield fixed-size batches from speaker-balanced indices."""

    def __init__(
        self,
        speakers: Sequence[int],
        batch_size: int,
        drop_last: bool = True,
        seed: int = 42,
    ):
        self.speakers = [int(s) for s in speakers]
        self.batch_size = int(batch_size)
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        indices = speaker_balanced_indices(self.speakers, self.epoch, self.seed)
        batch: list[int] = []
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        n = len(self.speakers)
        if self.drop_last:
            return n // self.batch_size
        return math.ceil(n / self.batch_size)
