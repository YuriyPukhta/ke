# RunPod Kokoro-82M LibriTTS fine-tuning

All training/launch code for RunPod lives here. **`third_party/` is never modified** — StyleTTS2 StyleTTS2 is imported read-only.

## Quick start (RunPod A40)

`runpod_training/payload/` already contains NPZ caches, phonemes, manifests, and QC indexes (**no WAV**).

```bash
cd /workspace/kokoro   # repo root with runpod_training/ + third_party/
export PYTHONPATH=$PWD
export WANDB_API_KEY=...   # optional secret
export DATA_ROOT=/workspace/data/LibriTTS_R   # WAV only
export RUN_ROOT=/workspace/kokoro-run

bash runpod_training/scripts/launch_runpod.sh
```

Upload to the pod:
1. Whole `runpod_training/` (includes `payload/`, ~185 MB data)
2. `third_party/StyleTTS2/` (StyleTTS2 dependency)
3. Separately: `LibriTTS_R` audio under `$DATA_ROOT`


Stages: index QC → convert weights → Stage1 (aligner warm-up + acoustic) → Stage2 (prosody + joint) → export KModel + voicepacks.

Resume:

```bash
RESUME=auto STAGE=stage1 bash runpod_training/scripts/launch_runpod.sh
RESUME=auto STAGE=stage2 bash runpod_training/scripts/launch_runpod.sh
```

## Layout

```text
runpod_training/
  configs/config_libritts_a40.yml
  data/prepare_index.py npz_dataset.py sampling.py
  train/train_stage1.py train_stage2.py freeze.py schedulers.py ...
  scripts/launch_runpod.sh preflight.py convert_kokoro_weights.py ...
  docs/RUNPOD.md
```

Volume artifacts: `/workspace/kokoro-run/{checkpoints,logs,metrics,samples,exports,state}`.

## Docs

See [docs/RUNPOD.md](docs/RUNPOD.md) for paths, LR schedule, freeze policy, and monitoring.
