#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
LIBERO_HOME="${LIBERO_HOME:-/workspace/tingting/LIBERO}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/workspace/kaixi/.conda/envs/libero-ttd/bin/python}"   # has robosuite/LIBERO; starVLA env does NOT
CKPT="${CKPT:-${STARVLA_DIR}/playground/Checkpoints/geo_memoryvla_dual_0630_0521/checkpoints/steps_30000_pytorch_model.pt}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-6694}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_spatial}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-10}"
MUJOCO_GL_VALUE="${MUJOCO_GL_VALUE:-osmesa}"   # EGL broken on this machine; osmesa = CPU software render
PYOPENGL_PLATFORM_VALUE="${PYOPENGL_PLATFORM_VALUE:-osmesa}"

if [[ -z "${LIBERO_HOME}" ]]; then
  echo "LIBERO_HOME is required."
  echo "Example: LIBERO_HOME=/path/to/LIBERO LIBERO_PYTHON=/path/to/python bash $0"
  exit 1
fi

cd "${STARVLA_DIR}"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${PYTHONPATH:-}:${LIBERO_HOME}:${STARVLA_DIR}"
export MUJOCO_GL="${MUJOCO_GL_VALUE}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM_VALUE}"

FOLDER_NAME="$(echo "${CKPT}" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')"
MODEL_ROOT="$(echo "${CKPT}" | awk -F'/checkpoints/' '{print $1}')"
VIDEO_OUT_PATH="${MODEL_ROOT}/results/${TASK_SUITE_NAME}/${FOLDER_NAME}"

"${LIBERO_PYTHON}" ./examples/LIBERO/eval_files/eval_libero.py \
  --args.pretrained-path "${CKPT}" \
  --args.host "${HOST}" \
  --args.port "${PORT}" \
  --args.task-suite-name "${TASK_SUITE_NAME}" \
  --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
  --args.video-out-path "${VIDEO_OUT_PATH}"
