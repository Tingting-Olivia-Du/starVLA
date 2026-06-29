# D4RT-WorldState — Code, Pipeline & Usage

**Date:** 2026-06-30
**Branch:** `d4rt-worldstate` (pushed to `myfork` = github.com/Tingting-Olivia-Du/starVLA)
**Spec:** [docs/superpowers/specs/2026-06-30-d4rt-worldstate-design.md](superpowers/specs/2026-06-30-d4rt-worldstate-design.md)
**Plan:** [docs/superpowers/plans/2026-06-30-d4rt-worldstate.md](superpowers/plans/2026-06-30-d4rt-worldstate.md)

---

## 1. What this is

A **second geometric world-state + forecaster stream** for starVLA's `GeoMemoryVLA`, using
**D4RT** instead of **VGGT + VGGT-World**. It is a *drop-in alternative* selected by one config
switch, so you can run a **controlled head-to-head comparison** — everything else (Qwen3-VL
semantic stream, dual memory bank, GR00T action head, dataloader, eval) is held constant.

```
                     framework.world_state.backbone
                    ┌──────────────┴──────────────┐
            "vggt_world" (arm A)              "d4rt" (arm B)
        VGGT-1B + VGGT-World flow head    Open-D4RT encoder + decoder forecaster
                    └──────────────┬──────────────┘
                          (identical downstream)
          Qwen3-VL  →  Dual Memory  →  Condition Assembler  →  GR00T head  →  actions
```

**Research bet:** D4RT has *native spatio-temporal correspondence*; turning it into a forecaster
(query its decoder at `t_tgt > t`) should beat a static encoder + bolted-on flow head.

> **Key fact:** D4RT (arXiv 2512.08924) is a **monocular reconstruction** model — it does NOT
> forecast out of the box (`t_tgt ∈ {1..T}`, observed frames only). Forecasting is **added** here
> by querying its 144M decoder at `t_tgt > t`. The encoder is frozen.

---

## 2. Environment

Use the **existing starVLA conda env** — do NOT make a new one, and do NOT install Open-D4RT's
`requirements.txt`.

```bash
STARVLA_PY=/workspace/ghsun/miniconda3/envs/starVLA/bin
export PATH="${STARVLA_PY}:${PATH}"
export PYTHONPATH=/workspace/tingting/starVLA
export HF_HUB_ENABLE_HF_TRANSFER=0      # hf_transfer pkg not installed in this env
```

| Why not a new env / not their requirements.txt | |
| --- | --- |
| The D4RT stream must co-run with Qwen3-VL + GR00T + memory, all in the starVLA env. | One model, one env. |
| Open-D4RT pins **numpy==2.2.5**; starVLA has **numpy 1.26.4** (pytorch3d + vggt depend on it). | Installing it breaks the env. |
| The vendored D4RT *model code* has no exotic deps (no transformers/timm/flash_attn) and runs under numpy 1.26. | Verified. |

---

## 3. Pretrained checkpoint (one-time download)

The vendored code needs the Open-D4RT **reproduction** checkpoint (Apache-2.0; unofficial,
"not endorsed by the original authors" — disclose in any publication).

**Use `OpenD4RT_48CLIP_9Mix_NoCropAUG`** (48-frame; more temporal context for forecasting,
matches the paper's training regime). `opend4rt.ckpt` is **14 GB**.

```bash
cd /workspace/tingting/starVLA
HF_HUB_ENABLE_HF_TRANSFER=0 ${STARVLA_PY}/huggingface-cli download \
  Lijiaxin0111/OpenD4RT --include "checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/*" \
  --local-dir playground/Pretrained_models/_opend4rt_dl

# place where the config expects it:
mkdir -p playground/Pretrained_models/OpenD4RT_48CLIP_9Mix_NoCropAUG
ln -sf /workspace/tingting/starVLA/playground/Pretrained_models/_opend4rt_dl/checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt \
       playground/Pretrained_models/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt
```

> The 14 GB ckpt is **not** in git. The 32-frame variant (`OpenD4RT_32CLIP_9Dataset_NoAUG`) is a
> fallback if you hit memory pressure — swap `clip_frames: 32` + its `model.yaml`.

---

## 4. Code map — what was added / changed

All new code is greppable with the marker `[D4RT-WorldState]`. Vendored code notes provenance
`Vendored from Open-D4RT (Apache-2.0)`.

| File | Status | Role |
| --- | --- | --- |
| `starVLA/model/modules/d4rt/` | **VENDORED** | Open-D4RT `src/model` + `src/core` (encoder, decoder, query embedding, heads, builder, config, checkpoint). `from src.` imports rewritten to the vendored package. |
| `starVLA/model/modules/d4rt/loader.py` | **NEW** | `build_d4rt_model(model_yaml, ckpt_path)` → `D4RTModel` (loads cfg + ckpt `strict=False`). |
| `starVLA/model/modules/d4rt/model_yaml_48.yaml` | **NEW** | Vendored 48-frame `model.yaml` (vit-g, clip_frames=48). |
| `starVLA/model/modules/geomem/d4rt_world_state_adapter.py` | **NEW** | `D4RTWorldStateAdapter` — frozen encoder. `.encode(window)→D4RTState`, `.flatten()→[B,N,1280]`, `.hidden_size=1280`. Mirrors `WorldStateAdapter`. |
| `starVLA/model/modules/geomem/d4rt_imagination_adapter.py` | **NEW** | `D4RTImaginationAdapter` — forecaster. `subgoal_type ∈ {latent, tracks}`. `.imagine_tokens()`, `.training_loss()`. Mirrors `ImaginationAdapter`. |
| `starVLA/model/modules/geomem/world_state_factory.py` | **NEW** | `build_world_state(fw)` / `build_imaginer(fw, ws)` — dispatch on `world_state.backbone`. |
| `starVLA/model/framework/VLM4A/GeoMemoryVLA.py` | **MODIFIED** | Factory switch; `_build_d4rt_window` (agentview-only temporal window); backbone-aware routing in `_encode`, `forward`, `predict_action`. |
| `examples/LIBERO/train_files/config_geomemvla_d4rt.yaml` | **NEW** | D4RT-arm config (controlled copy of the geomemvla config; only `world_state`/`imagination` differ). |
| `tools/d4rt_forecast_probe.py` | **NEW** | Phase D-PROBE: offline forecasting probe (`build_forecast_query`, `probe_extrapolation`). |
| `tests/geomemvla/test_d4rt_*.py` | **NEW** | 6 unit test files (loader, adapters, probe, window, config, factory) + 1 slow GPU smoke. |

### Two design invariants baked into the code
- **B1 — agentview-only (monocular).** D4RT has no multi-view axis, so the geometric stream
  consumes ONLY the agentview camera as a *temporal* window (`_build_d4rt_window`). The wrist
  camera still reaches the policy via the Qwen3-VL/GR00T path. (D4RT memory C=1280 vs VGGT 1024
  — `geo_dim` adapts automatically via `.hidden_size`.)
- **`t_cam = t_tgt`** pinned in every forecasting query (`build_forecast_query`) — future
  positions expressed in the frame the policy acts from.

---

## 5. Data flow (the D4RT arm)

```
examples (LIBERO batch)
   │  e["image"] = [agentview, wrist]   ;   e["image_window"] = List[F][views]  (when imag on)
   ▼
_build_d4rt_window(examples)            # agentview (view 0) only, TIME axis, [B,F,3,256,256], /255
   ▼
D4RTWorldStateAdapter.encode(window)    # frozen D4RT.encode_video → memory [B,N,1280] = D4RTState
   │
   ├─► world-state stream  geo = state.flatten() = memory  ───────────┐
   │                                                                   │
   └─► D4RTImaginationAdapter (when imagination on)                    │
         subgoal_type="latent": LatentForecastHead(memory) → [B,N,1280]│
         subgoal_type="tracks": decode_queries(t_tgt>t) → xyz[B,M,3]   │
                                 → Linear(3→1280) → [B,M*horizon,1280]  │
         training_loss: latent=MSE(F̂, memory); tracks=L1(xyz, GT)      │
                                                                        ▼
                Dual Memory Bank → Condition Assembler → GR00T head → action chunk
```

---

## 6. How to call it

### 6a. Quick sanity (no GPU, no ckpt) — unit tests
```bash
cd /workspace/tingting/starVLA
${STARVLA_PY}/python -m pytest tests/geomemvla/ -m "not slow" -q
# expect: 61 passed
```

### 6b. Phase D0 — end-to-end smoke on GPU (needs the 14 GB ckpt)
Proves the D4RT encoder + GR00T head train end-to-end. **Use GPUs 4–7.**
```bash
cd /workspace/tingting/starVLA
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/workspace/tingting/starVLA HF_HUB_ENABLE_HF_TRANSFER=0 \
  ${STARVLA_PY}/python -m pytest tests/geomemvla/test_d4rt_smoke_gpu.py -v -m slow -s
# expect: action_loss finite, predict_action shape (1,8,7)
```

### 6c. Phase D-PROBE — forecasting gate (run BEFORE training `tracks`)
Measures whether the frozen D4RT decoder can extrapolate `t_tgt > t` on real LIBERO clips.
If the MAE is poor, ship `subgoal_type=latent` only.
```python
# minimal driver (run inside the starVLA env, GPU 4):
import torch
from starVLA.model.modules.d4rt.loader import build_d4rt_model
from tools.d4rt_forecast_probe import probe_extrapolation

model = build_d4rt_model(
    "starVLA/model/modules/d4rt/model_yaml_48.yaml",
    "playground/Pretrained_models/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt",
    device="cuda",
)
clip = load_libero_agentview_clip()          # [1, T, 3, 256, 256] in [0,1]
uv   = end_effector_and_grid_points()        # [M, 2] in [0,1]
out  = probe_extrapolation(model, clip, uv, prefix_len=T-2, horizon=2)
print("forecast MAE:", out["mae"])           # low = decoder extrapolates → proceed to tracks
```

### 6d. Training — pick the arm with one flag

The framework reads `framework.world_state.backbone`. Two equivalent ways:

**(i) Use the D4RT config directly:**
```bash
cd /workspace/tingting/starVLA
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH=/workspace/tingting/starVLA HF_HUB_ENABLE_HF_TRANSFER=0 \
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/LIBERO/train_files/config_geomemvla_d4rt.yaml \
  --run_root_dir ./playground/Checkpoints \
  --run_id geomemvla_d4rt_$(date +%m%d_%H%M)
```

**(ii) Override the existing geomemvla launcher** (`run_geo_memoryvla_train.sh`) by adding the
backbone/ckpt flags — every config key is a CLI override:
```bash
  --framework.world_state.backbone d4rt \
  --framework.world_state.d4rt_model_yaml starVLA/model/modules/d4rt/model_yaml_48.yaml \
  --framework.world_state.d4rt_ckpt_path playground/Pretrained_models/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt \
  --framework.world_state.clip_frames 48 \
  --framework.imagination.subgoal_type latent
```

### 6e. Eval / inference
The eval path is **unchanged** — `predict_action` is backbone-agnostic (it routes
`_build_d4rt_window` internally when `backbone==d4rt`). Point your existing LIBERO eval
(`examples/LIBERO/eval_files/eval_libero.sh`) at a D4RT checkpoint dir.

---

## 7. The comparison matrix (ablation switches)

| Switch | Values | Meaning |
| --- | --- | --- |
| `framework.world_state.backbone` | `vggt_world` \| `d4rt` | **the arm being compared** |
| `framework.world_state.stream` | `geo_only` \| `sem_only` \| `dual` | which streams feed the policy |
| `framework.memory.enabled` | bool | dual memory bank on/off |
| `framework.imagination.enabled` | bool | forecaster on/off |
| `framework.imagination.subgoal_type` | `latent` \| `tracks` | **D4RT only** (vggt_world ignores it) |

**Phase plan** (each = a measurable row):
- **D0** — D4RT encoder only (imagination off). Integration proof. ✅ verified
- **D1** — `latent` forecaster. Apples-to-apples vs VGGT-World imagination. ✅ E2E verified
- **D2** — `tracks` forecaster (D4RT-native future 3D points). ✅ E2E verified
- **D3** — full matrix `{vggt_world,d4rt} × {no-imag,latent,tracks} × {geo_only,dual}`.

---

## 8. Verified status & known follow-ups

**Verified on GPU 4 with the real 14 GB ckpt:** D0, D1 (latent), D2 (tracks) all produce finite
`action_loss` + `imagination_loss` + `predict_action (1,8,7)`. 61 unit tests pass; no regression
to the VGGT-World path.

**Deferred (per plan — do before the D2/D3 training runs):**
1. **D-PROBE manual gate** — measure real extrapolation MAE before trusting `tracks`.
2. **`tracks` supervision** — the loss is a **no-op when `gt_future_xyz` is absent**; wire the
   full-clip GT-future positions from the dataloader. End-effector pixel selection is currently a
   uniform grid (`track_grid`), refine to the actual gripper pixel.
3. **Reproduction-encoder caveat (R5)** — Open-D4RT weights are unofficial/WIP; a weak encoder
   bounds the comparison. The D-PROBE transitively measures this.
```
