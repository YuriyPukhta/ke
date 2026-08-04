"""NPZ-backed LibriTTS dataset compatible with StyleTTS2 collate tuple."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from runpod_training.data.sampling import SpeakerBalancedBatchSampler

logger = logging.getLogger(__name__)

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300
)
MEL_MEAN, MEL_STD = -4.0, 4.0


def preprocess(wave: np.ndarray) -> torch.Tensor:
    wave_tensor = torch.from_numpy(np.asarray(wave, dtype=np.float32))
    mel = to_mel(wave_tensor)
    mel = (torch.log(1e-5 + mel.unsqueeze(0)) - MEL_MEAN) / MEL_STD
    return mel


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class NPZTargetsDataset(Dataset):
    """Cache-faithful audio (no ±5000 pad) + exact Kokoro phoneme_ids from NPZ."""

    def __init__(
        self,
        index_jsonl: str | Path,
        data_root: str | Path,
        targets_root: str | Path | None = None,
        ood_texts: str | Path | None = None,
        sr: int = 24000,
        min_length: int = 50,
        validation: bool = False,
        use_cached_f0n: bool = False,
        mel_len_tol: int = 1,
        max_mel_length: int = 192,
    ):
        self.rows = load_jsonl(Path(index_jsonl))
        self.data_root = Path(data_root)
        self.targets_root = Path(targets_root) if targets_root else None
        self.sr = sr
        self.min_length = min_length
        self.validation = validation
        self.use_cached_f0n = use_cached_f0n
        self.mel_len_tol = mel_len_tol
        self.max_mel_length = max_mel_length
        self.skipped = 0

        self.ood_texts: list[list[int]] = []
        if ood_texts and Path(ood_texts).is_file():
            # Prefer phoneme_ids from matching index rows when available; else skip OOD.
            for line in Path(ood_texts).read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                # OOD file stores IPA strings; we map via a random train row's
                # phoneme_ids of similar length only for shape — StyleTTS2 uses
                # TextCleaner. For exact Kokoro IDs we sample from this index.
                pass
        # Build OOD candidate id sequences from this split (deterministic refs).
        self.ood_candidates = [
            r["phoneme_ids"] for r in self.rows if len(r.get("phoneme_ids", [])) >= min_length
        ]
        if not self.ood_candidates:
            self.ood_candidates = [r["phoneme_ids"] for r in self.rows if r.get("phoneme_ids")]

        # Speaker -> row indices for reference sampling
        self.by_speaker: dict[int, list[int]] = {}
        for i, row in enumerate(self.rows):
            spk = int(row["speaker_id"])
            self.by_speaker.setdefault(spk, []).append(i)

    def __len__(self) -> int:
        return len(self.rows)

    def speakers(self) -> list[int]:
        return [int(r["speaker_id"]) for r in self.rows]

    def _resolve_audio(self, row: dict[str, Any]) -> Path:
        for key in ("audio_path", "audio"):
            value = row.get(key)
            if not value:
                continue
            path = Path(str(value))
            if path.is_file():
                return path
            cand = self.data_root / value
            if cand.is_file():
                return cand
        raise FileNotFoundError(f"audio missing for {row.get('id')}: {row.get('audio')}")

    def _resolve_npz(self, row: dict[str, Any]) -> Path:
        for key in ("npz_path", "cache"):
            value = row.get(key)
            if not value:
                continue
            path = Path(str(value))
            if path.is_file():
                return path
            if self.targets_root is not None:
                cand = self.targets_root / value
                if cand.is_file():
                    return cand
            # fallback: next to package payload
            cand = Path(__file__).resolve().parents[1] / "payload" / "kokoro_targets_v1" / value
            if cand.is_file():
                return cand
        raise FileNotFoundError(f"npz missing for {row.get('id')}: {row.get('cache')}")

    def _load_wave(self, rel_or_abs: str | Path) -> np.ndarray:
        path = Path(rel_or_abs)
        if not path.is_file():
            path = self.data_root / rel_or_abs
        wave, sr = sf.read(path.as_posix(), always_2d=False)
        if getattr(wave, "ndim", 1) > 1:
            wave = wave[:, 0]
        wave = np.asarray(wave, dtype=np.float32)
        if sr != self.sr:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=self.sr).astype(np.float32)
        # Cache-faithful: NO ±5000 StyleTTS2 pad
        return wave

    def _ids_tensor(self, ids: list[int]) -> torch.Tensor:
        # BOS/EOS pad index 0
        return torch.LongTensor([0, *[int(x) for x in ids], 0])

    def _ref_index(self, speaker_id: int, self_idx: int) -> int:
        pool = self.by_speaker.get(speaker_id, [self_idx])
        if self.validation:
            # Deterministic: first other utterance, else self
            for j in pool:
                if j != self_idx:
                    return j
            return self_idx
        choices = [j for j in pool if j != self_idx] or pool
        return int(np.random.choice(choices))

    def _ood_text(self) -> torch.Tensor:
        if not self.ood_candidates:
            return torch.LongTensor([0, 0])
        ids = self.ood_candidates[int(np.random.randint(0, len(self.ood_candidates)))]
        # Ensure min_length in tokens
        while len(ids) < self.min_length and len(self.ood_candidates) > 1:
            extra = self.ood_candidates[int(np.random.randint(0, len(self.ood_candidates)))]
            ids = list(ids) + list(extra)
            if len(ids) > 510:
                ids = ids[:510]
                break
        return self._ids_tensor(ids[:510])

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        try:
            wave = self._load_wave(self._resolve_audio(row))
            mel = preprocess(wave).squeeze()
            mel_len = mel.size(1)
            mel = mel[:, : mel_len - (mel_len % 2)]
            mel_len = mel.size(1)

            expected = int(row["mel_len"])
            if abs(mel_len - expected) > self.mel_len_tol:
                raise ValueError(
                    f"mel_len mismatch for {row['audio']}: got {mel_len}, cache {expected}"
                )

            text = self._ids_tensor(row["phoneme_ids"])
            if text.numel() > 512:
                raise ValueError(f"text too long: {text.numel()}")

            speaker_id = int(row["speaker_id"])
            ref_idx = self._ref_index(speaker_id, idx)
            ref_row = self.rows[ref_idx]
            ref_wave = self._load_wave(self._resolve_audio(ref_row))
            ref_mel = preprocess(ref_wave).squeeze()
            ref_mel_len = ref_mel.size(1)
            if ref_mel_len > self.max_mel_length:
                start = 0 if self.validation else int(np.random.randint(0, ref_mel_len - self.max_mel_length))
                ref_mel = ref_mel[:, start : start + self.max_mel_length]
            elif ref_mel_len % 2 == 1:
                ref_mel = ref_mel[:, :-1]

            ref_text = self._ood_text()
            cached_f0 = cached_n = None
            npz_path = self._resolve_npz(row)
            if self.use_cached_f0n:
                with np.load(npz_path, allow_pickle=False) as data:
                    cached_f0 = torch.from_numpy(np.asarray(data["f0"], dtype=np.float32))
                    cached_n = torch.from_numpy(np.asarray(data["n"], dtype=np.float32))
                    # Align to even mel length
                    cached_f0 = cached_f0[:mel_len]
                    cached_n = cached_n[:mel_len]

            sample = (
                speaker_id,
                mel,
                text,
                ref_text,
                ref_mel,
                int(ref_row["speaker_id"]),
                row["audio"],
                wave,
            )
            # Sidecar attached via attribute for optional collate use
            meta = {
                "cached_f0": cached_f0,
                "cached_n": cached_n,
                "d_gt": row.get("phoneme_ids") and None,  # QC only by default
                "align_score": float(row.get("align_score", 0.0)),
                "mel_len": mel_len,
            }
            return sample, meta
        except Exception as exc:
            self.skipped += 1
            logger.warning("skip %s: %s", row.get("audio"), exc)
            # Fallback to another index to keep DataLoader alive
            alt = (idx + 1) % max(len(self.rows), 1)
            if alt == idx:
                raise
            return self[alt]


class NPZCollater:
    def __init__(self, return_wave: bool = True, use_cached_f0n: bool = False):
        self.max_mel_length = 192
        self.return_wave = return_wave
        self.use_cached_f0n = use_cached_f0n

    def __call__(self, batch):
        # batch items are (sample_tuple, meta)
        samples = [b[0] for b in batch]
        metas = [b[1] for b in batch]
        batch_size = len(samples)
        lengths = [b[1].shape[1] for b in samples]
        order = np.argsort(lengths)[::-1]
        samples = [samples[i] for i in order]
        metas = [metas[i] for i in order]

        nmels = samples[0][1].size(0)
        max_mel_length = max(b[1].shape[1] for b in samples)
        max_text_length = max(b[2].shape[0] for b in samples)
        max_rtext_length = max(b[3].shape[0] for b in samples)

        labels = torch.zeros(batch_size).long()
        mels = torch.zeros(batch_size, nmels, max_mel_length).float()
        texts = torch.zeros(batch_size, max_text_length).long()
        ref_texts = torch.zeros(batch_size, max_rtext_length).long()
        input_lengths = torch.zeros(batch_size).long()
        ref_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()
        ref_mels = torch.zeros(batch_size, nmels, self.max_mel_length).float()
        ref_labels = torch.zeros(batch_size).long()
        waves = [None] * batch_size
        paths = [""] * batch_size

        f0s = ns = None
        if self.use_cached_f0n:
            f0s = torch.zeros(batch_size, max_mel_length).float()
            ns = torch.zeros(batch_size, max_mel_length).float()

        for bid, (label, mel, text, ref_text, ref_mel, ref_label, path, wave) in enumerate(samples):
            mel_size = mel.size(1)
            text_size = text.size(0)
            rtext_size = ref_text.size(0)
            labels[bid] = label
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            ref_texts[bid, :rtext_size] = ref_text
            input_lengths[bid] = text_size
            ref_lengths[bid] = rtext_size
            output_lengths[bid] = mel_size
            paths[bid] = path
            ref_mel_size = min(ref_mel.size(1), self.max_mel_length)
            ref_mels[bid, :, :ref_mel_size] = ref_mel[:, :ref_mel_size]
            ref_labels[bid] = ref_label
            waves[bid] = wave
            if self.use_cached_f0n and metas[bid]["cached_f0"] is not None:
                f0s[bid, :mel_size] = metas[bid]["cached_f0"][:mel_size]
                ns[bid, :mel_size] = metas[bid]["cached_n"][:mel_size]

        # StyleTTS2-compatible primary tuple
        primary = (
            waves,
            texts,
            input_lengths,
            ref_texts,
            ref_lengths,
            mels,
            output_lengths,
            ref_mels,
        )
        if self.use_cached_f0n:
            return primary + (f0s, ns, paths)
        return primary + (paths,)


def build_npz_dataloader(
    index_jsonl: str | Path,
    data_root: str | Path,
    batch_size: int = 12,
    num_workers: int = 8,
    validation: bool = False,
    ood_texts: str | Path | None = None,
    targets_root: str | Path | None = None,
    use_cached_f0n: bool = False,
    speaker_balanced: bool = True,
    seed: int = 42,
) -> DataLoader:
    dataset = NPZTargetsDataset(
        index_jsonl=index_jsonl,
        data_root=data_root,
        targets_root=targets_root,
        ood_texts=ood_texts,
        validation=validation,
        use_cached_f0n=use_cached_f0n,
    )
    collate = NPZCollater(use_cached_f0n=use_cached_f0n)
    if validation or not speaker_balanced:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=not validation,
            num_workers=num_workers,
            drop_last=not validation,
            collate_fn=collate,
            pin_memory=True,
        )
    sampler = SpeakerBalancedBatchSampler(dataset.speakers(), batch_size=batch_size, seed=seed)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
    )
