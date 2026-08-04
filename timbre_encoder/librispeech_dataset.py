"""LibriTTS_R / LibriSpeech-style dataset for dual style encoder training."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Optional

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from gt_cache import DEFAULT_FEATURES_ROOT, feature_path, load_curve
from .mel_preprocess import SAMPLE_RATE, waveform_to_mel

ROOT = Path(__file__).resolve().parents[1]


def text_to_kokoro_phonemes(text: str, pipeline) -> str:
    phonemes = ""
    for _, ps, _ in pipeline.en_tokenize(pipeline.g2p(text)[1]):
        phonemes += ps
    return phonemes


def parse_d_gt(s: str | None) -> Optional[list[int]]:
    if not s or not str(s).strip():
        return None
    try:
        return [int(x) for x in str(s).split()]
    except ValueError:
        return None


class LibriTTSDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str | Path,
        pipeline,
        max_phonemes: int = 400,
        min_phonemes: int = 2,
        max_wav_sec: float = 8.0,
        device: str = "cpu",
        require_alignment: bool = False,
        load_f0n: bool = True,
        require_f0n: bool = False,
        features_root: str | Path | None = None,
        speaker_ids: set[str] | list[str] | None = None,
    ):
        self.pipeline = pipeline
        self.max_phonemes = max_phonemes
        self.min_phonemes = min_phonemes
        self.max_wav_sec = max_wav_sec
        self.device = device
        self.require_alignment = require_alignment
        self.load_f0n = load_f0n
        self.require_f0n = require_f0n
        self.features_root = Path(features_root) if features_root else DEFAULT_FEATURES_ROOT
        allowed = {str(s) for s in speaker_ids} if speaker_ids else None
        self.rows: list[dict[str, str]] = []
        with open(manifest_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if allowed is not None and str(row.get("speaker_id", "")) not in allowed:
                    continue
                if require_alignment and row.get("align_status") != "ok":
                    continue
                if require_f0n and (
                    row.get("f0_status") != "ok" or row.get("n_status") != "ok"
                ):
                    continue
                self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Optional[dict[str, Any]]:
        row = self.rows[idx]
        # Prefer cached raw Kokoro phonemes; fall back to G2P
        phonemes = (row.get("phonemes") or "").strip()
        if not phonemes:
            try:
                phonemes = text_to_kokoro_phonemes(row["text"], self.pipeline)
            except Exception:
                return None
        phonemes_st2 = (row.get("phonemes_st2") or "").strip()
        # Legacy: old cache stored ST2-normalized string in `phonemes`
        if not phonemes_st2 and row.get("norm_applied") not in ("0", "1"):
            legacy_raw = (row.get("phonemes_raw") or "").strip()
            if legacy_raw:
                phonemes_st2 = phonemes
                phonemes = legacy_raw
        if not (self.min_phonemes <= len(phonemes) <= self.max_phonemes):
            return None
        d_gt = parse_d_gt(row.get("d_gt"))
        if self.require_alignment and not d_gt:
            return None
        # d_gt is always for ST2 phonemes when present
        align_phones = phonemes_st2 or phonemes
        if d_gt is not None and len(d_gt) != len(align_phones):
            return None
        wav, _ = librosa.load(row["audio_path"], sr=SAMPLE_RATE, mono=True)
        wav, _ = librosa.effects.trim(wav, top_db=30)
        max_samples = int(self.max_wav_sec * SAMPLE_RATE)
        if wav.shape[0] > max_samples:
            wav = wav[:max_samples]
        if wav.shape[0] < SAMPLE_RATE // 4:
            return None
        wave_t = torch.from_numpy(wav.astype(np.float32))
        mel = waveform_to_mel(wave_t)  # [1, 80, T]
        mel_2d = mel.squeeze(0)  # [80, T]
        item: dict[str, Any] = {
            "audio_path": row["audio_path"],
            "text": row["text"],
            "speaker_id": row["speaker_id"],
            "phonemes": phonemes,
            "waveform": wave_t,
            "mel": mel_2d,
        }
        if phonemes_st2:
            item["phonemes_st2"] = phonemes_st2
        if row.get("norm_applied") in ("0", "1"):
            item["norm_applied"] = row["norm_applied"] == "1"
        if d_gt is not None:
            item["d_gt"] = torch.tensor(d_gt, dtype=torch.long)
        if self.load_f0n:
            f0 = load_curve(feature_path(row["audio_path"], "f0", self.features_root))
            n = load_curve(feature_path(row["audio_path"], "n", self.features_root))
            t_mel = int(mel_2d.shape[-1])
            # Cache uses max_wav_sec=8; train may use a shorter cap — same prefix trim.
            if f0 is not None and f0.numel() >= t_mel:
                item["f0"] = f0[:t_mel]
            elif self.require_f0n:
                return None
            if n is not None and n.numel() >= t_mel:
                item["n"] = n[:t_mel]
            elif self.require_f0n:
                return None
        return item



def collate_skip_none(batch: list[Optional[dict]]) -> Optional[dict]:
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    # Train loop uses batch_size=1 for aligner simplicity by default;
    # still support list return for bs=1.
    if len(batch) == 1:
        return batch[0]
    return batch
