#!/usr/bin/env bash
# [D4RT-WorldState] LIBERO training launcher: D4RT as the geometric world model.
# stream=dual (D4RT geo + Qwen3-VL sem + memory + imagination=latent). GPUs 0,1.
# subgoal_type is locked to `latent` in the config (D-PROBE gate FAILED for `tracks`;
# frozen D4RT decoder does not extrapolate — see docs/D4RT_WORLDSTATE_PIPELINE.md).
# See docs/superpowers/specs/2026-06-30-d4rt-worldstate-design.md
set -euo pipefail

# ---- environment ----------------------------------------------------------
STARVLA_PY=/workspace/ghsun/miniconda3/envs/starVLA/bin
export PATH="${STARVLA_PY}:${PATH}"
export PYTHONPATH=/workspace/tingting/starVLA
export NO_ALBUMENTATIONS_UPDATE=1
export HF_HUB_ENABLE_HF_TRANSFER=0
# [D4RT-WorldState] reduce allocator fragmentation (OOM log recommended this). geo/imag are
# now spatially pooled (geo_pool_tokens), so condition is ~386 tokens not 12290 — peak mem
# ~25 GiB at bs=4 (was 124 GiB -> OOM). This is belt-and-suspenders against fragmentation.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# [D4RT-WorldState] GPUs 0,1 per user request (override by exporting CUDA_VISIBLE_DEVICES).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# [Geo-MemoryVLA] /tmp is a SMALL 30G disk — route temp/scratch to the big rootfs disk.
BIGTMP=/workspace/tingting/.bigtmp
mkdir -p "${BIGTMP}" "${BIGTMP}/torch" "${BIGTMP}/triton" "${BIGTMP}/wandb-cache"
export TMPDIR="${BIGTMP}"
export HF_HOME="${HF_HOME:-/root/angli/hf_cache}"
export TORCH_HOME="${BIGTMP}/torch"
export TRITON_CACHE_DIR="${BIGTMP}/triton"
export WANDB_CACHE_DIR="${BIGTMP}/wandb-cache"
export WANDB_DIR="${BIGTMP}"

cd /workspace/tingting/starVLA

# ---- config ---------------------------------------------------------------
config_yaml=./examples/LIBERO/train_files/config_geomemvla_d4rt.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all
run_root_dir=./playground/Checkpoints
run_id=d4rt_worldmodel_dual_$(date +%m%d_%H%M)   # pass --run_id to override for resume
stream=dual                                       # D4RT geo + Qwen3-VL sem

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/" 2>/dev/null || true

num_processes=${NUM_PROCESSES:-$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)}
export WANDB_MODE="${WANDB_MODE:-online}"
wandb_entity="${WANDB_ENTITY:-tingtingdu06-uw-madison}"

# [D4RT-WorldState] Pick an EXPLICIT free master port. NOTE: accelerate treats
# `--main_process_port 0` as the literal port 0 (NOT "auto-pick"), which makes every rank
# hang forever in _create_c10d_store (NCCL rendezvous). The default 29500 is also occupied
# by other jobs here. So choose a concrete free port (override with MAIN_PROCESS_PORT).
main_port="${MAIN_PROCESS_PORT:-29555}"
if ss -tln 2>/dev/null | grep -q ":${main_port} "; then
  # fall back: scan upward for the first free port so two D4RT runs don't collide.
  for p in $(seq 29556 29600); do
    ss -tln 2>/dev/null | grep -q ":${p} " || { main_port=$p; break; }
  done
fi
echo "[D4RT-WorldState] using master port ${main_port}"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2_d4rt.yaml \
  --num_processes "${num_processes}" \
  --main_process_port "${main_port}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name GeoMemoryVLA \
  --framework.world_state.backbone d4rt \
  --framework.world_state.stream ${stream} \
  --framework.world_state.d4rt_model_yaml starVLA/model/modules/d4rt/model_yaml_48.yaml \
  --framework.world_state.d4rt_ckpt_path playground/Pretrained_models/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt \
  --framework.world_state.clip_frames 48 \
  --framework.memory.enabled True \
  --framework.imagination.enabled True \
  --framework.imagination.subgoal_type latent \
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
  --wandb_project starVLA_D4RT_WorldModel \
  --wandb_entity "${wandb_entity}"
