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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"

# [Geo-MemoryVLA] /tmp is a SMALL 30G disk — temp files there cause ENOSPC mid-train.
# Route ALL temp/scratch to the big rootfs disk (/, == /workspace, 5T). run_root_dir is already
# under /workspace below. NOTE: HF cache already lives on the big disk at /root/angli/hf_cache
# (VGGT-1B 5G is cached there) — point HF_HOME at it so we DON'T re-download. Only the pure
# temp paths (tmpfile / triton / wandb scratch) are redirected to BIGTMP.
BIGTMP=/workspace/tingting/.bigtmp
mkdir -p "${BIGTMP}" "${BIGTMP}/torch" "${BIGTMP}/triton" "${BIGTMP}/wandb-cache"
export TMPDIR="${BIGTMP}"
export HF_HOME="${HF_HOME:-/root/angli/hf_cache}"   # existing big-disk cache (VGGT-1B already here)
export TORCH_HOME="${BIGTMP}/torch"
export TRITON_CACHE_DIR="${BIGTMP}/triton"
export WANDB_DIR="${BIGTMP}"
export WANDB_CACHE_DIR="${BIGTMP}/wandb-cache"

cd /workspace/tingting/starVLA

# ---- config ---------------------------------------------------------------
config_yaml=./examples/LIBERO/train_files/geo_memoryvla_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all              # start small (1 dataset); switch to libero_all for full
run_root_dir=./playground/Checkpoints
run_id=geo_memoryvla_dual_$(date +%m%d_%H%M)   # NOTE: pass --run_id to override for resume

# GEO-ONLY: no Qwen3-VL. stream=geo_only, imagination+memory ON. 2. sem_only 3. dual
stream=dual

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/" 2>/dev/null || true

num_processes=${NUM_PROCESSES:-$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)}

# WANDB: enabled by default (entity tingtingdu). Export WANDB_MODE=disabled to skip;
# make sure you're logged in (`wandb login`) or export WANDB_API_KEY.
export WANDB_MODE="${WANDB_MODE:-online}"
wandb_entity="${WANDB_ENTITY:-tingtingdu06-uw-madison}"

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
  --datasets.vla_data.per_device_batch_size 8 \
  --datasets.vla_data.sequential_step_sampling True \
  --datasets.vla_data.enable_image_window True \
  --trainer.max_train_steps 80000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 50 \
  --trainer.eval_interval 1000 \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project starVLA_GeoMemoryVLA \
  --wandb_entity "${wandb_entity}"
