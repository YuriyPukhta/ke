#!/usr/bin/env bash
# Validate volume after attaching to a new RunPod. See /workspace/kokoro/POD_RESTORE.md
set -euo pipefail

WS="${WS:-/workspace}"
REPO="${REPO:-${WS}/kokoro}"
RUN_ROOT="${RUN_ROOT:-${WS}/kokoro-run}"

echo "=== Pod restore check ==="
echo "workspace: ${WS}"
df -h "${WS}" | tail -1 || true

fail=0
need_dir() {
  if [[ -d "$1" ]]; then
    echo "OK  dir  $1"
  else
    echo "MISS dir  $1"
    fail=1
  fi
}
need_file() {
  if [[ -f "$1" ]]; then
    echo "OK  file $1"
  else
    echo "MISS file $1"
    fail=1
  fi
}
warn_dir() {
  if [[ -d "$1" ]]; then
    echo "OK  dir  $1"
  else
    echo "WARN dir $1"
  fi
}

need_dir "${REPO}/runpod_training"
need_dir "${REPO}/runpod_training/payload/kokoro_targets_v1"
need_dir "${REPO}/third_party/kikiri-tts/StyleTTS2"

# Audio: either layout
if [[ -d "${WS}/kokoro_assets/data/LibriTTS_R" ]]; then
  echo "OK  audio ${WS}/kokoro_assets/data/LibriTTS_R"
  DATA_ROOT="${DATA_ROOT:-${WS}/kokoro_assets/data/LibriTTS_R}"
elif [[ -d "${WS}/data/LibriTTS_R" ]]; then
  echo "OK  audio ${WS}/data/LibriTTS_R"
  DATA_ROOT="${DATA_ROOT:-${WS}/data/LibriTTS_R}"
else
  echo "MISS audio under kokoro_assets/data/LibriTTS_R or data/LibriTTS_R"
  fail=1
  DATA_ROOT="${DATA_ROOT:-}"
fi

warn_dir "${RUN_ROOT}"
warn_dir "${RUN_ROOT}/checkpoints/stage1"
warn_dir "${RUN_ROOT}/checkpoints/stage2"
warn_dir "${RUN_ROOT}/metrics"
warn_dir "${RUN_ROOT}/logs/tensorboard"
warn_dir "${RUN_ROOT}/logs/console"
warn_dir "${RUN_ROOT}/cache"

echo
echo "--- state pointers ---"
for f in latest.json latest_stage1.json latest_stage2.json; do
  p="${RUN_ROOT}/state/${f}"
  if [[ -f "$p" ]]; then
    echo "--- ${f} ---"
    cat "$p"
  else
    echo "(no ${f})"
  fi
done

echo
echo "--- NPZ count (may take a bit) ---"
npz_n=$(find "${REPO}/runpod_training/payload/kokoro_targets_v1" -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
echo "npz files: ${npz_n}"

echo
echo "Suggested env:"
echo "  export REPO=${REPO}"
echo "  export PYTHONPATH=${REPO}"
echo "  export RUN_ROOT=${RUN_ROOT}"
[[ -n "${DATA_ROOT}" ]] && echo "  export DATA_ROOT=${DATA_ROOT}"
echo "  export SKIP_INDEX=1"
[[ -f "${RUN_ROOT}/weights/kokoro_base.pth" ]] && echo "  export SKIP_CONVERT=1"
echo "  RESUME=auto STAGE=stage1 bash ${REPO}/runpod_training/scripts/launch_runpod.sh"

if [[ "${fail}" -ne 0 ]]; then
  echo
  echo "RESTORE CHECK FAILED — see POD_RESTORE.md"
  exit 1
fi
echo
echo "RESTORE CHECK OK"
