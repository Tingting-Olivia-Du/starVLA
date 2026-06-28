#!/usr/bin/env bash
# [Geo-MemoryVLA] Real LIBERO training launcher for the GeoMemoryVLA framework.
# Defaults to the GEO-ONLY ablation (VGGT 3D world-state + memory + imagination,
# NO Qwen3-VL) — this exercises the genuinely-new chain without the ~8GB Qwen3-VL
# download. Run on GPUs 0-3. See docs/superpowers/plans/2026-06-29-geo-memoryvla-*.md
set -euo pipefail

# ---- environment ----------------------------------------------------------
# Use the starVLA conda env (has pytorch3d + vggt + transformers 4.57).
STARVLA_PY=/workspace/ghsun/miniconda3/envs/starVLA/bin
export PATH="${STARVLA_PY}:${PATH}"
export PYTHONPATH=/workspace/tingting/starVLA
export NO_ALBUMENTATIONS_UPDATE=1
export HF_HUB_ENABLE_HF_TRANSFER=0
# GPUs 0-3 (current convention). Override by exporting CUDA_VISIBLE_DEVICES before calling.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

cd /workspace/tingting/starVLA

# ---- config ---------------------------------------------------------------
config_yaml=./examples/LIBERO/train_files/geo_memoryvla_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_goal              # start small (1 dataset); switch to libero_all for full
run_root_dir=./playground/Checkpoints
run_id=geo_memoryvla_geoonly_$(date +%m%d_%H%M)   # NOTE: pass --run_id to override for resume

# GEO-ONLY: no Qwen3-VL. stream=geo_only, imagination+memory ON.
stream=geo_only

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/" 2>/dev/null || true

num_processes=${NUM_PROCESSES:-$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)}

# WANDB: set WANDB_MODE=disabled to skip, or export WANDB_API_KEY + entity/project.
export WANDB_MODE="${WANDB_MODE:-disabled}"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_processes}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name GeoMemoryVLA \
  --framework.world_state.stream ${stream} \
  --framework.memory.enabled True \
  --framework.imagination.enabled True \
  --datasets.vla_data.data_root_dir "${libero_data_root}" \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 4 \
  --datasets.vla_data.sequential_step_sampling True \
  --datasets.vla_data.enable_image_window True \
  --trainer.max_train_steps 80000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 50 \
  --trainer.eval_interval 1000 \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project starVLA_GeoMemoryVLA \
  --wandb_entity your_name
