# Geo-MemoryVLA — Design Spec

**Date:** 2026-06-29
**Framework host:** starVLA (`/workspace/tingting/starVLA`)
**Status:** Design approved, pending implementation plan

---

## 0. Coding convention (applies to ALL code in this project)

Every new file and every modified block introduced for this project MUST be tagged so it
is greppable later. Use the marker `[Geo-MemoryVLA]`.

- **New file header:**
  ```python
  # [Geo-MemoryVLA] <one-line role of this file in the design>
  # Part of the Geo-MemoryVLA architecture — see
  # docs/superpowers/specs/2026-06-29-geo-memoryvla-design.md
  ```
- **Modified block in an existing file:**
  ```python
  # [Geo-MemoryVLA] <what was added/changed and why>
  ```
- When code is lifted from MemoryVLA (MIT) or VGGT-World, note the provenance in the
  header (`# Adapted from MemoryVLA (MIT): vla/memory_vla.py:CogMemBank`).

---

## 1. Research thesis

> **When a VLA's memory and imagination both operate over a 3D-geometry latent space
> (VGGT) in addition to a 2D-semantic latent space (VLM), do long-horizon,
> spatially-grounded manipulation tasks improve over a purely-2D MemoryVLA++?**

Each component has validated prior work; the novelty is moving memory + imagination into a
**dual (geometric + semantic) latent space** where the geometric stream uses a VGGT
*world-state* representation:

| Building block | Prior work that validates it |
| --- | --- |
| VGGT frozen features help manipulation policies | **GeoAware-VLA** (arXiv 2509.14117): +35pt on unseen LIBERO viewpoints |
| VGGT latent as a world-state you can roll forward in time | **VGGT-World** (arXiv 2603.12655): frozen VGGT layer-4 latent + flow transformer predicting geometry evolution |
| memory bank + imagination + diffusion action loop | **MemoryVLA / MemoryVLA++** (arXiv 2508.19236 / 2606.09827) |

**Open gap (the contribution):** nobody has coupled a VGGT *world-state* (VGGT-World, which
is action-free) with a *policy* that has *dual-latent memory + 3D imagination*. That is
Geo-MemoryVLA.

---

## 2. Approved decisions (locked)

| Decision | Choice |
| --- | --- |
| Imagination scope | **Full three-piece**: dual memory + 3D imagination + action head |
| Memory representation | **Dual memory**: VGGT geometric latent (perceptual) + VLM semantic latent (cognitive) |
| Language / semantic model | **Qwen3-VL small** (already wired in starVLA; provides the semantic memory stream + language grounding) |
| Action head | **starVLA GR00T dual-system flow-matching head** (reuse starVLA training loop / eval / ckpt unchanged) |
| VGGT | **Frozen** feature extractor (layer-4 latent, per VGGT-World) |

---

## 3. Architecture & data flow

```
current obs images ─┬─→ [FROZEN VGGT, layer-4]  ──→ z_t   (3D geometric latent)   [B, N_g, 1024]
                    └─→ [Qwen3-VL]              ──→ s_t   (2D semantic cog token) [B, 1, H]
                                                   + perceptual vision tokens     [B, N_v, H]
                                                   + language tokens
        ┌───────────────────────────────────────────────────────────────────────┘
        ▼
  Dual Memory Bank   (MemoryVLA CogMemBank/PerMemBank, lifted)
     ├─ geometric memory bank : stores history z_{t-k}      ─→ retrieve m_geo
     └─ semantic  memory bank : stores history s_{t-k}      ─→ retrieve m_sem
                                (cross-attn retrieval + gate fusion + ToMe consolidation)
        │
        ▼
  3D Imagination   (VGGT-World flow transformer, z-prediction, new module)
     └─ imagine future ẑ_{t+1..t+m} in the VGGT latent space  ← the "3D visual subgoal"
        │
        ▼
  Fuse condition = [ z_t , m_geo , s_t , m_sem , ẑ_subgoal , language tokens ]
        │   (project all streams to GR00T cross-attention dim, concat along sequence)
        ▼
  GR00T dual-system flow-matching action head  (starVLA, unchanged)
        └─→ action chunk  [B, action_horizon, action_dim]
```

**Why `z` unifies your three intuitions:** the VGGT latent `z` simultaneously serves as (a)
the current *world state*, (b) the unit stored in *geometric memory*, and (c) the object
*imagined* as a future *3D visual subgoal*. One representation, three roles.

---

## 4. Components

Files live under `starVLA/starVLA/model/`. Reuse verdicts come from the MemoryVLA /
beta-vla / starVLA code audits.

### 4.1 VGGT world-state encoder — vendored from VGGT-World (REVISED)
- **Use the real VGGT-World implementation, not a hand-written re-encoder.** The reference
  is at `/workspace/tingting/vggt-world/vggt/world_model/` (Meta research license; allows
  derivative works + research use, requires acknowledgement in publications).
- Vendor that whole `world_model/` package into starVLA at
  `starVLA/model/modules/vggt_world/` (verbatim, headers tagged + provenance noted).
- The world state is `FrozenVGGTBackbone.encode_states(images) -> GeometryState`
  (`backbone.py:84`). It runs VGGT **layers 0-5** and returns a **dual-stream**
  `GeometryState` (`frame_tokens` + `global_tokens`, each `[B, frames, tokens, 1024]`,
  plus `patch_grid`, `frame_ids`, `image_hw`). This supersedes the earlier single-`z`
  assumption — the real world state is two token streams, decoder-compatible.
- For downstream consumers (memory, condition assembler) flatten via
  `GeometryState.flatten_streams() -> [B, frames*2*tokens, 1024]` (`state.py:55`).
- **Verified:** the installed `vggt` package exposes everything `backbone.py` imports
  (`slice_expand_and_flatten`, `Aggregator._process_frame_attention/_process_global_attention`),
  so the vendored package runs without patching VGGT itself.
- **Reuse:** directly liftable (vendor verbatim); only a thin starVLA-facing adapter is new.

### 4.2 Dual memory bank — `modules/memory/dual_memory_bank.py` (NEW, ~250 LOC, core research)
- Two instances of MemoryVLA's bank (lifted, MIT):
  - `geo_bank`  : token dim = 1024 (VGGT latent)
  - `sem_bank`  : token dim = H (Qwen3-VL hidden) for cognitive token; optional perceptual sub-bank for vision tokens
- Mechanisms reused verbatim (dim-agnostic): cross-attention retrieval (`retrieval_layers`),
  gate fusion (`fusion_type='gate'`), **ToMe** redundancy consolidation
  (`consolidate_type='tome'`), timestep positional encoding.
- **Episode/timestep bookkeeping:** MemoryVLA's bank is keyed by `episode_id` + `timestep`.
  starVLA's dataloader must surface these per sample. **Open implementation task** (see §7):
  verify the starVLA LIBERO dataloader exposes episode id + timestep, or add them.
- **Reuse:** `CogMemBank` / `PerMemBank` from `MemoryVLA/vla/memory_vla.py:158-358` —
  directly liftable, change dims. `BottleneckSE` (`:105-137`) rewritten for VGGT feature shape.

### 4.3 3D imagination — vendored `VGGTWorldModel` (REVISED)
- **Use the real VGGT-World flow transformer, not a hand-written toy.** It lives in the same
  vendored package (§4.1): `vggt_world/model.py:VGGTWorldModel`,
  `flow_model.py:TemporalFlowTransformer`, `losses.py:WorldModelLoss`, `solver.py`,
  `scheduler.py`. Faithful to the paper because it *is* the paper's code.
- Real mechanism (now confirmed by reading the code):
  - **Architecture:** FLUX-style `DoubleStreamBlock` + `SingleStreamBlock` with **3-axis RoPE**
    (`EmbedND3D`, axes `[16,24,24]` over frame / patch-h / patch-w), not a generic encoder.
  - **z-prediction:** the flow net predicts the clean target; sampling converts to velocity via
    `solver.clean_to_velocity` inside an Euler rollout.
  - **stage-1 loss:** SNR-weighted MSE `1/(1-τ)²` (`losses.py:_latent_loss`); **stage-2:** plain
    MSE on the flow-forced target.
  - **two-stage flow-forcing curriculum:** `_stage1_forward` (teacher-forced) and
    `_stage2_forward` (partial Euler rollout + `mix_lambda` linear ramp), gated by `where`
    vs `stage2_start` (`model.py:159-242`).
  - **autoregressive forecast:** chunk-wise sliding-window `_forecast` with `context_size`/
    `chunk_size` (`model.py:256`).
  - **bonus:** `decode_during_train` can add geometry-decode supervision (depth/point/camera)
    on top of the latent loss (`backbone.decode_geometry`), exactly as the paper does.
- **Geo-MemoryVLA wiring:** a thin adapter calls `VGGTWorldModel` to produce imagined future
  `GeometryState` (the 3D visual subgoal), then `flatten_streams()` into the condition. The
  imaginer's own loss (latent + optional decode) is added to the action loss with a scale.
- **Supervision (free):** GT future state = run `FrozenVGGTBackbone.encode_states` on the
  trajectory's future frames; LIBERO trajectories are available offline. The model's
  `forward(images, ...)` already encodes the full window and slices context/target internally.
- **Reuse:** directly liftable (vendor verbatim); only the starVLA-facing adapter + the
  future-frame batching are new.

### 4.4 Semantic / language stream — Qwen3-VL (REUSE, no change)
- Use starVLA's `_QWen3_VL_Interface` (`modules/vlm/QWen3.py`) unchanged.
- Cognitive token = `hidden_states[-1]` at the last text position; perceptual tokens = the
  vision-token slice of `hidden_states[-1]`. Both already available via
  `output_hidden_states=True`.

### 4.5 Action head — GR00T dual-system (REUSE, no change)
- `modules/action_model/GR00T_ActionHeader.py`, exactly as `QwenGR00T` uses it.
- The head consumes a single `last_hidden`-like `[B, L, H]` condition + optional `state`.
  Geo-MemoryVLA builds that `[B, L, H]` by projecting and concatenating all streams (§3),
  so **the head is untouched**.

### 4.6 Framework orchestrator — `framework/VLM4A/GeoMemoryVLA.py` (NEW, ~250 LOC)
- Skeleton copied from `framework/VLM4A/QwenGR00T.py` (same `forward` / `predict_action`
  contract, same GR00T head wiring).
- Adds: VGGT encoder, dual memory bank, imaginer; assembles the fused condition; routes to
  GR00T head. Registered via `@FRAMEWORK_REGISTRY.register("GeoMemoryVLA")`.
- Default-config dataclass `GeoMemoryVLADefaultConfig` (pattern from `QwenGR00TDefaultConfig`).

### 4.7 Wiring / config (SMALL edits)
- `modules/vlm/__init__.py`: no change needed (we don't route VGGT through `get_vlm_model`;
  the framework instantiates VGGT directly). Keep the semantic path on Qwen3-VL.
- `examples/LIBERO/train_files/config_geomemvla.yaml` (NEW, ~60 LOC): copy an existing
  GR00T LIBERO config; add `framework.world_state`, `framework.memory`, `framework.imagination`
  sections.

---

## 5. Control variant (mandatory)

To attribute any eval result correctly, the design includes an **ablation switch** in config
so the same framework can run reduced variants without code forks:

- `memory.enabled` (on/off)
- `imagination.enabled` (on/off)
- `world_state.stream` ∈ {`geo_only`, `sem_only`, `dual`}

This yields the key baselines for the paper:
1. **Qwen3-VL + GR00T** (no memory, no imagination, sem only) — the starVLA `QwenGR00T` baseline.
2. **+ dual world-state** (no memory, no imagination).
3. **+ dual memory**.
4. **+ 3D imagination** (full Geo-MemoryVLA).

This also operationalizes the `eval≈0` debugging discipline from prior sessions: a reduced
variant that matches `QwenGR00T` proves the harness/data path is intact, isolating any
failure to the new modules.

---

## 6. Implementation phasing (within the full three-piece scope)

The scope is the full system; phasing is purely to de-risk and to produce intermediate
ablation points — not to descope.

- **Phase A — World-state baseline:** VGGT encoder + project `z_t` into the GR00T condition
  alongside Qwen3-VL. No memory, no imagination. Proves VGGT-as-world-state plugs into the
  GR00T head and trains. (≈ variant 2.)
- **Phase B — Dual memory:** add lifted dual memory bank. (≈ variant 3.)
- **Phase C — 3D imagination:** add the flow imaginer; supervise on future VGGT latents;
  stage-1 (teacher-forced) first, stage-2 curriculum as an ablation. (≈ variant 4, full system.)

Each phase is independently trainable and gives a measurable ablation row.

---

## 7. Open implementation tasks / risks (to resolve during planning)

1. **Episode id + timestep in dataloader.** MemoryVLA's bank requires per-sample
   `episode_id` + `timestep`. Confirm starVLA's LIBERO dataloader provides them; if not, add
   them. **Blocking for Phase B.**
2. **Geometry vs semantics in retrieval.** Pure VGGT geometry is weak on object semantics
   ("the red cup"). This is exactly why memory is *dual* — semantic recall rides the Qwen3-VL
   stream. Validate that the semantic bank actually carries object identity.
3. **Future-VGGT supervision cost.** Phase C needs frozen-VGGT latents for future frames.
   Decide: precompute offline (faster training, more disk) vs on-the-fly (less disk, slower).
4. **Token budget.** Dual streams + memory + imagined subgoal inflate the GR00T condition
   sequence length. Set `num_target_vision_tokens` / projection pooling to keep it bounded.
5. **VGGT input layout.** VGGT expects multi-view frames with camera/register tokens; map
   LIBERO's agentview + wrist cameras to VGGT's `num_cameras` convention (beta-vla config
   uses `num_cameras=2`, matching LIBERO dual-cam).
6. **Compute.** Frozen VGGT-1B + Qwen3-VL + GR00T head is heavy; per memory note, use GPUs
   4-7 for smoke runs.

---

## 8. Code-change magnitude

| Type | Files | Est. LOC |
| --- | --- | --- |
| VENDORED verbatim (VGGT-World) | `modules/vggt_world/*` (backbone, flow_model, model, losses, solver, scheduler, state, blocks, rope3d, time_embed, metrics) | ~1577 (copied, not written) |
| NEW research module | `dual_memory_bank.py` (+ `memory_bank.py`) | ~300 (the real new research code) |
| NEW wiring/adapters | `vggt_world_adapter.py`, `condition_assembler.py`, `GeoMemoryVLA.py` (self-registers — no `__init__.py` edit), `geo_memoryvla_libero.yaml` | ~500 (mostly skeletons/glue) |
| REUSE unchanged | Qwen3-VL wrapper, GR00T head, starVLA train loop / eval / ckpt | 0 |
| Lifted from MemoryVLA (MIT) | memory bank, ToMe, gate fusion | adapt only |

**Rating:** vendoring VGGT-World removes the biggest research risk (the imaginer is now the
paper's own code, not a re-derivation). Genuinely new code shrinks to the dual memory bank +
the glue that turns `GeometryState` into the GR00T condition. Heavier than a static VGGT+GR00T
model, but a faithful, citable paper story.

> **Correction note (2026-06-29):** an earlier draft planned to *re-implement* the imaginer
> from the paper abstract. After locating the real VGGT-World code at
> `/workspace/tingting/vggt-world`, the design switched to **vendoring it verbatim** — the
> hand-written `GeoFlowImaginer` toy is dropped. This was prompted by the user correctly
> challenging whether a from-abstract reimplementation would match the paper. It would not have.

---

## 9. Provenance & licenses

- **MemoryVLA** — MIT. Memory bank / ToMe / gate fusion are liftable; cite in headers.
- **VGGT** (`facebook/VGGT-1B`) — frozen pretrained encoder via the `vggt` package (installed,
  v0.0.1). **License caveat:** the original `VGGT-1B` checkpoint is **non-commercial**; a
  separate `VGGT-1B-Commercial` checkpoint exists for commercial use. Fine for research; note
  before any productization.
- **VGGT-World** (arXiv 2603.12655) — implementation **vendored verbatim** from
  `/workspace/tingting/vggt-world/vggt/world_model/` under the **Meta research license**
  (permits derivative works + research use; **requires acknowledgement in publications**).
  Verified that the installed `vggt` package satisfies its internal imports.
- **starVLA** — host framework (MIT); GR00T head, Qwen3-VL wrapper, trainer reused.
