"""Data package exports (lazy-safe)."""

__all__ = [
    "NPZTargetsDataset",
    "build_npz_dataloader",
    "SpeakerBalancedBatchSampler",
]


def __getattr__(name: str):
    if name in {"NPZTargetsDataset", "build_npz_dataloader"}:
        from runpod_training_encoders.data.npz_dataset import NPZTargetsDataset, build_npz_dataloader

        return {"NPZTargetsDataset": NPZTargetsDataset, "build_npz_dataloader": build_npz_dataloader}[name]
    if name == "SpeakerBalancedBatchSampler":
        from runpod_training_encoders.data.sampling import SpeakerBalancedBatchSampler

        return SpeakerBalancedBatchSampler
    raise AttributeError(name)
