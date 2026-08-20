#!/usr/bin/env bash
# Launch full Kokoro LibriTTS pipeline on RunPod (A40).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PKG="${ROOT}/runpod_training"
PAYLOAD="${PKG}/payload"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="${CONFIG:-${PKG}/configs/config_libritts_a40.yml}"
RUN_ROOT="${RUN_ROOT:-/workspace/kokoro-run}"
DATA_ROOT="${DATA_ROOT:-/workspace/data/LibriTTS_R}"
RESUME="${RESUME:-}"
SKIP_INDEX="${SKIP_INDEX:-1}"   # indexes already in payload by default
SKIP_CONVERT="${SKIP_CONVERT:-0}"
STAGE="${STAGE:-all}"  # all|index|stage1|stage2|export

mkdir -p "${RUN_ROOT}"/{checkpoints/stage1,checkpoints/stage2,logs/tensorboard,metrics,samples,exports/kmodel,exports/voicepacks,configs,state,indexes,weights}

# Persist runtime config and rewrite data roots for this machine
python - <<PY
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path(r"""${CONFIG}""").read_text(encoding="utf-8"))
cfg["run_root"] = r"""${RUN_ROOT}"""
cfg["pretrained_model"] = str(Path(r"""${RUN_ROOT}""") / "weights" / "kokoro_base.pth")
cfg["first_stage_path"] = str(Path(r"""${RUN_ROOT}""") / "checkpoints" / "stage1" / "first_stage.pth")
cfg.setdefault("data_params", {})
cfg["data_params"]["root_path"] = r"""${DATA_ROOT}"""
cfg["data_params"]["targets_root"] = str(Path(r"""${PAYLOAD}""") / "kokoro_targets_v1")
cfg["data_params"]["train_index"] = str(Path(r"""${PAYLOAD}""") / "indexes" / "train.jsonl")
cfg["data_params"]["val_index"] = str(Path(r"""${PAYLOAD}""") / "indexes" / "dev.jsonl")
cfg["data_params"]["OOD_data"] = str(Path(r"""${PAYLOAD}""") / "indexes" / "OOD_texts.txt")
out = Path(r"""${RUN_ROOT}""") / "configs" / "runtime.yml"
out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print("wrote", out)
PY

echo "==> Preflight"
python -m runpod_training.scripts.preflight --config "${RUN_ROOT}/configs/runtime.yml"

if [[ ! -d "${PAYLOAD}/kokoro_targets_v1" ]]; then
  echo "ERROR: missing ${PAYLOAD}/kokoro_targets_v1 — ship runpod_training/payload with the package"
  exit 1
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "ERROR: missing audio root ${DATA_ROOT} (WAV only; not in payload)"
  exit 1
fi

if [[ "${SKIP_CONVERT}" != "1" ]]; then
  echo "==> Convert Kokoro weights"
  python -m runpod_training.scripts.convert_kokoro_weights \
    --output "${RUN_ROOT}/weights/kokoro_base.pth"
fi

if [[ "${SKIP_INDEX}" != "1" && ( "${STAGE}" == "all" || "${STAGE}" == "index" ) ]]; then
  echo "==> Prepare indexes"
  python -m runpod_training.data.prepare_index \
    --train-csv "${TRAIN_CSV:-${PAYLOAD}/manifests/train-clean-100.filtered.csv}" \
    --dev-csv "${DEV_CSV:-${PAYLOAD}/manifests/dev-clean.filtered.csv}" \
    --phonemes-jsonl "${PHONEMES_JSONL:-${PAYLOAD}/phonemes_libritts_filtered.jsonl}" \
    --targets-manifest "${TARGETS_MANIFEST:-${PAYLOAD}/kokoro_targets_v1/manifest.jsonl}" \
    --targets-root "${TARGETS_ROOT:-${PAYLOAD}/kokoro_targets_v1}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${PAYLOAD}/indexes" \
    --min-align-score "${MIN_ALIGN_SCORE:-0.25}"
fi

# TensorBoard in background
if command -v tensorboard >/dev/null 2>&1; then
  echo "==> TensorBoard on :6006"
  tensorboard --logdir "${RUN_ROOT}/logs/tensorboard" --host 0.0.0.0 --port 6006 >/tmp/tensorboard.log 2>&1 &
fi

RESUME_ARGS=()
if [[ -n "${RESUME}" ]]; then
  RESUME_ARGS=(--resume "${RESUME}")
fi

if [[ "${STAGE}" == "all" || "${STAGE}" == "stage1" ]]; then
  echo "==> Stage 1"
  python -m runpod_training.train.train_stage1 --config "${RUN_ROOT}/configs/runtime.yml" "${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}"
fi

if [[ "${STAGE}" == "all" || "${STAGE}" == "stage2" ]]; then
  echo "==> Stage 2"
  python -m runpod_training.train.train_stage2 --config "${RUN_ROOT}/configs/runtime.yml" --resume "${RESUME:-auto}"
fi

if [[ "${STAGE}" == "all" || "${STAGE}" == "export" ]]; then
  echo "==> Export"
  python -m runpod_training.scripts.export_kmodel \
    --checkpoint "${RUN_ROOT}/checkpoints/stage2/final.pth" \
    --output "${RUN_ROOT}/exports/kmodel/kokoro_libritts.pth" \
    --config-json "${RUN_ROOT}/weights/config.json"
  python -m runpod_training.scripts.extract_voicepacks \
    --checkpoint "${RUN_ROOT}/checkpoints/stage2/final.pth" \
    --style-checkpoint "${RUN_ROOT}/checkpoints/stage1/first_stage.pth" \
    --index "${PAYLOAD}/indexes/train.jsonl" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${RUN_ROOT}/exports/voicepacks" \
    --speakers "${VOICE_SPEAKERS:-103,1188,1272}"
fi

echo "==> Done. Artifacts under ${RUN_ROOT}"
