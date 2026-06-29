# D4RT-WorldState — Design Spec

**Date:** 2026-06-30
**Framework host:** starVLA (`/workspace/tingting/starVLA`)
**Status:** Design approved, pending implementation plan
**Companion spec:** [`2026-06-29-geo-memoryvla-design.md`](2026-06-29-geo-memoryvla-design.md) (the VGGT-World variant this is compared against)

---

## 0. Coding convention (applies to ALL code in this project)

Every new file and modified block MUST be greppable. Use the marker `[D4RT-WorldState]`.

- **New file header:**
  ```python
  # [D4RT-WorldState] <one-line role of this file in the design>
  # Part of the D4RT-WorldState comparison arm — see
  # docs/superpowers/specs/2026-06-30-d4rt-worldstate-design.md
  ```
- **Modified block in an existing file:**
  ```python
  # [D4RT-WorldState] <what was added/changed and why>
  ```
- When code is vendored from Open-D4RT (Apache-2.0), note provenance in the header
  (`# Vendored from Open-D4RT (Apache-2.0): github.com/Lijiaxin0111/Open-d4rt`).

---

## 1. Research thesis & the controlled experiment

> **A model with native spatio-temporal correspondence (D4RT), turned into a forecaster by
> extending its query time-axis to `t_tgt > t`, produces a better manipulation subgoal than a
> static encoder + bolted-on flow head (VGGT + VGGT-World) — with everything else held constant.**

The contribution is a **head-to-head comparison**, not a new policy from scratch. The existing
Geo-MemoryVLA pipeline (memory + semantic stream + GR00T head + dataloader + eval) is reused
verbatim; the **only** independent variable is the geometric world-state + forecaster stream.

### Controlled experiment

One config switch `framework.world_state.backbone ∈ {vggt_world, d4rt}` selects the stream.
Everything else is byte-identical.

| Held constant (shared) | Varied (the independent variable) |
| --- | --- |
| Qwen3-VL semantic stream, dual memory bank, GR00T flow-matching head, LIBERO dataloader, eval harness, ablation switches, action config | **A (`vggt_world`):** VGGT-1B (frozen) + VGGT-World flow head (trained from scratch) — *the existing design* <br> **B (`d4rt`):** Open-D4RT encoder (frozen, Apache-2.0 ckpt) + D4RT forecaster head |

Any eval delta between A and B is attributable to the 4D substrate + forecaster, because
nothing else differs.

---

## 2. Verified facts about D4RT (paper 2512.08924 + Open-D4RT code)

These facts drive every design decision below; they were verified against the paper HTML and
the Open-D4RT repo, not assumed.

| Fact | Value | Source | Design consequence |
| --- | --- | --- | --- |
| Encoder | ViT-g, 40 layers, ~1B params, **VideoMAE-initialized** (not VGGT) | paper §arch | Frozen backbone; no VGGT lineage. |
| Decoder | tiny ~144M cross-attention head | paper §arch | **Forecasting surgery touches only this.** |
| Query | `q = (u, v, t_src, t_tgt, t_cam)` + 9×9 RGB patch → 3D point in cam frame `t_cam` | paper §query | Explicit, disentangled time axis → forecasting = extend `t_tgt`. |
| Input | **monocular video** `V ∈ ℝ^{T×H×W×3}`, ONE image per timestep | paper §3, verified | **No view axis.** Forces B1 (agentview-only). See §5. |
| `t_cam` | selects *which frame's pose* is the reference coord system, NOT which camera | paper Fig.2 | Pin `t_cam = t` (current frame). See Risk 2. |
| Temporal patch | 2×16×16 (needs a real temporal clip) | paper §arch | Adapter feeds a 32/48-frame sliding window. See Risk 3. |
| Forecasting | **NOT in the model**: `t_tgt ∈ {1…T}`, observed frames only | paper, verified ×2 | The one real research bet; de-risked by a standalone probe first. See §4 + Risk 1. |
| Checkpoints | Open-D4RT, **Apache-2.0**, HF `Lijiaxin0111/OpenD4RT/checkpoints` — two variants (27.9 GB total): `OpenD4RT_32CLIP_9Dataset_NoAUG` and `OpenD4RT_48CLIP_9Mix_NoCropAUG` (newer; the one we target) | repo / HF | Vendored verbatim. **Unofficial *reproduction* weights**, "not endorsed by original authors"; README does NOT report a reproduction-vs-paper accuracy gap and full train/eval code is "planned for release" (WIP). Quality risk, not blocking — see R5. Disclosed. |

**Contrast with VGGT-World (verified in the vendored code):** VGGT's backbone has a single
`frames` axis (`vggt_world/backbone.py:43`) that is **view-agnostic** — used as *views* in the
world-state path (`_build_vggt_pixels`, `[B, num_views, 3, H, W]`) but as *time* in the
imagination path (`_build_image_window`, primary-camera temporal window). D4RT has only a time
axis. Crucially, **the existing imagination path is ALREADY single-camera-temporal**, so B1
introduces no asymmetry in the imagination/forecasting comparison — only in the static
world-state encode (VGGT 2 views vs D4RT 1 view), which is a documented footnote.

---

## 3. Architecture — two new adapters, identical interfaces

The existing orchestrator (`framework/VLM4A/GeoMemoryVLA.py`) consumes the world-state stream
through two adapter interfaces. D4RT implements the **same two interfaces**, so the orchestrator
changes by a ~30-LOC factory switch and **ConditionAssembler / DualMemoryBank / Qwen3-VL /
GR00T head / dataloader / eval change by ZERO**.

```
agentview temporal window ─→ [FROZEN D4RT encoder, ViT-g] ──→ F   (Global Scene Repr.)   [B, N, C]
        │                                                     │
        │   (shared, unchanged)                               ▼
        │                                          D4RT Forecaster (decoder fine-tune)
        │                                          ├─ subgoal_type="latent": F̂_{t+k}
        │                                          └─ subgoal_type="tracks": future 3D
        │                                             positions of gripper/grid points @ t_tgt>t
        ▼                                                     │
  Qwen3-VL semantic stream (UNCHANGED)                        ▼
        └────────────┬─────────── Dual Memory Bank (UNCHANGED) ───────────┐
                     ▼                                                     ▼
              ConditionAssembler (UNCHANGED) → GR00T flow-matching head (UNCHANGED) → action chunk
```

### 3a. `D4RTWorldStateAdapter` — mirrors `WorldStateAdapter`
File: `model/modules/geomem/d4rt_world_state_adapter.py` (NEW).
- Loads frozen Open-D4RT encoder (`OpenD4RT_48CLIP_9Mix_NoCropAUG`).
- `.encode(window) → D4RTState`: ViT-g encoder over the agentview temporal window → Global
  Scene Representation `F` (+ patch grid / frame ids kept for queries).
- `.flatten(state) → [B, N, C]` and `.hidden_size` — **drop-in** for memory + condition
  assembler (same contract as `WorldStateAdapter.encode/.flatten/.hidden_size`).
- **Frozen**; only a small projection to the GR00T condition dim is trained.

### 3b. `D4RTImaginationAdapter` — mirrors `ImaginationAdapter`
File: `model/modules/geomem/d4rt_imagination_adapter.py` (NEW).
Implements **both** subgoal types behind `imagination.subgoal_type ∈ {latent, tracks}` (the
approved "both as an ablation axis"):
- **`tracks`** (D4RT-native): query a fixed grid of 2D points + the end-effector pixel at
  `t_tgt = t+1…t+k` → **future 3D positions in camera space** = the literal "where the arm/object
  goes" subgoal. Trains by fine-tuning the **144M decoder only** to accept `t_tgt > t`.
- **`latent`** (parity with VGGT-World): a small forecaster head predicts `F̂_{t+k}`, fed as a
  latent subgoal — the apples-to-apples slot against VGGT-World's `ẑ`.
- Returns the subgoal **and its own forecasting loss** (added to action loss with
  `imagination.loss_scale`), exactly like the VGGT-World imaginer does today.

### 3c. Orchestrator factory switch (SMALL edit)
`framework/VLM4A/GeoMemoryVLA.py`: select `WorldStateAdapter`/`ImaginationAdapter` (vggt_world)
vs `D4RTWorldStateAdapter`/`D4RTImaginationAdapter` (d4rt) on `fw.world_state["backbone"]`.
~30 LOC; no other framework change.

---

## 4. The forecasting surgery (the one real research risk, named)

D4RT does NOT forecast out of the box. Forecasting is added by:
- **Freeze the 1B encoder** on observed frames `1…t`.
- **Fine-tune the 144M decoder** (+ its discrete timestep embedding) to extrapolate `t_tgt > t`.
- **Free supervision:** future frames are offline (LIBERO trajectories). Run **full D4RT** on the
  full clip → ground-truth future 3D positions; supervise the causal (prefix-only) forecaster
  against them. Same trick the VGGT-World stream uses (frozen-VGGT future latents as targets).

**Standalone de-risk probe (PHASE D-PROBE, runs BEFORE any VLA training):** feed frozen
Open-D4RT a LIBERO clip *prefix* (frames `1…t`), query gripper/grid points at `t_tgt = t+1, t+2`,
compare to the ground-truth positions D4RT gives on the *full* clip. Pure offline open-loop test
(no GR00T, no policy training). Mirrors the team's established open-loop-probe discipline
(see memory `pi05-openloop-probe-vs-fcdecoder`). **If the decoder can't extrapolate even after
light fine-tuning, fall back to `latent` subgoal and know it on day one.**

---

## 5. LIBERO multi-camera handling — B1 (agentview-only), locked

D4RT is monocular (§2). LIBERO has agentview (fixed external) + wrist (moving). Decision: the
**D4RT geometric stream consumes agentview-only** as its monocular temporal window. Rationale:
- A mostly-static external camera observing a dynamic arm+object scene is **D4RT's training
  distribution** — its sweet spot. The wrist is a violently-moving ego camera (its hard case).
- The **shared Qwen3-VL + GR00T path still sees both cameras**, so the policy is not blind to
  the wrist; only the *geometric* stream is single-view.
- The existing VGGT-World **imagination path is already single-camera-temporal**, so B1 creates
  no asymmetry in the forecasting comparison — only in the static encode (VGGT 2 views vs D4RT 1),
  a disclosed footnote, not a confound.

(Rejected: B2 two-pass concat — doubles compute, forces D4RT to solve violent wrist ego-motion;
B3 interleaved pseudo-video — breaks D4RT's monocular-motion assumption.)

---

## 6. Risk register & handling (locked)

| Risk | Handling |
| --- | --- |
| **R1 — Extrapolation may not generalize** (core bet) | **Standalone offline probe first** (§4 / Phase D-PROBE), before any VLA compute. Fall back to `latent` subgoal if tracks are unstable. |
| **R2 — Camera coords / `t_cam`** | **Pin `t_cam = t`** (current frame) as a hard adapter invariant, not a config knob. Future positions expressed in the frame the policy acts from; LIBERO agentview is ~fixed → stable metric frame. |
| **R3 — VideoMAE temporal patch needs a clip** | Adapter feeds a **sliding 32/48-frame agentview window** (matching the ckpt), built from frames the dataloader already loads; pad/stride at episode start; stride chosen so the window spans a meaningful motion span at LIBERO control freq. Contained entirely in `D4RTWorldStateAdapter`; shared dataloader unchanged. |
| **R4 — Compute (big backbones)** | Comparison runs are **A xor B**, never both in one model → peak memory ≈ existing design (D4RT-1B replaces VGGT-1B). Frozen + bf16; only 144M decoder + projections get gradients. Smoke on **GPUs 4–7**. |
| **R5 — Reproduction-grade encoder** (Open-D4RT ≠ original D4RT) | The HF weights are an **unofficial reproduction**, WIP, with **no published accuracy gap vs the paper**. A weaker encoder → weaker forecasts → smaller eval gains, but does NOT change the architecture, API, or forecasting surgery. **Quality risk, not blocking.** Phase D-PROBE transitively tests encoder quality (you cannot extrapolate well from a bad `F`), so it surfaces before VLA compute. If the probe is poor, the finding itself is reportable ("comparison bounded by reproduction encoder quality") and a paper-grade D4RT can be dropped in later as an encoder swap. |

---

## 7. Phasing (each phase = a measurable ablation row, de-risk order)

- **Phase D-PROBE — Forecasting probe.** Offline open-loop extrapolation test of frozen+lightly-
  finetuned Open-D4RT (§4). No GR00T. **Gate:** decoder extrapolates → proceed to D1; else pivot
  to `latent`-only.
- **Phase D0 — Encoder swap, no forecasting.** D4RT frozen encoder → condition (imagination off).
  Proves D4RT plugs into the shared pipeline and trains. Compare to Geo-MemoryVLA world-state
  baseline. *Kills integration risk before forecasting.*
- **Phase D1 — `latent` forecaster.** Add `F̂_{t+k}` subgoal. Apples-to-apples vs VGGT-World
  imagination.
- **Phase D2 — `tracks` forecaster.** Add explicit future-3D-point subgoal (D4RT's unique play).
  The headline result.
- **Phase D3 — Full comparison matrix.** {vggt_world, d4rt} × {no-imag, latent, tracks} ×
  {geo_only, dual} on LIBERO, one table.

Each phase is independently trainable and toggled by config.

---

## 8. Control variants / ablation switches (reuse existing + 2 new)

Reuse the existing Geo-MemoryVLA switches (`memory.enabled`, `imagination.enabled`,
`world_state.stream ∈ {geo_only, sem_only, dual}`). Add:
- `world_state.backbone ∈ {vggt_world, d4rt}` — selects the comparison arm.
- `imagination.subgoal_type ∈ {latent, tracks}` — only meaningful for `d4rt` (vggt_world is
  always latent-`ẑ`); for `vggt_world` it is ignored.

---

## 9. Code-change magnitude

| Type | Files | Est. |
| --- | --- | --- |
| NEW adapters | `geomem/d4rt_world_state_adapter.py`, `geomem/d4rt_imagination_adapter.py` | ~400 LOC |
| VENDORED verbatim | Open-D4RT encoder + decoder under `modules/d4rt/` (Apache-2.0) | copied |
| SMALL edit | `GeoMemoryVLA.py` factory switch on `world_state.backbone`; new `config_geomemvla_d4rt.yaml` | ~30 LOC |
| NEW (offline) | `tools/d4rt_forecast_probe.py` (Phase D-PROBE) | ~150 LOC |
| REUSE unchanged | ConditionAssembler, DualMemoryBank, Qwen3-VL, GR00T head, dataloader, eval, ablation switches | **0** |

---

## 10. Provenance & licenses

- **Open-D4RT** — Apache-2.0, `github.com/Lijiaxin0111/Open-d4rt`, ckpt HF `Lijiaxin0111/OpenD4RT`.
  **Unofficial** reproduction ("not affiliated with or endorsed by the original D4RT authors") —
  disclose in any publication; cite both Open-D4RT and the original paper (arXiv 2512.08924).
- **starVLA / Geo-MemoryVLA** — host pipeline reused unchanged (memory bank MIT-lifted from
  MemoryVLA, GR00T head, Qwen3-VL wrapper, trainer, eval). See companion spec for their licenses.
- **VGGT-World arm** — unchanged from the companion spec (VGGT-1B frozen; VGGT-World vendored
  under Meta research license). Only relevant as comparison baseline A.

---

## 11. Open implementation tasks (resolve during planning)

1. **Open-D4RT API surface.** Map the repo's `src/` model class to `.encode`/query calls;
   confirm exact input tensor shape `[B, T, 3, H, W]` and query signature. (Web docs didn't expose
   function names — read `src/` directly during planning.)
2. **Window/stride mapping.** Pick 32 vs 48 ckpt and a stride so the agentview window spans a
   meaningful motion horizon at LIBERO control frequency.
3. **Track point selection for `tracks`.** Define the grid + how the end-effector pixel is
   located each step (from proprio-projected gripper, or a fixed salient grid).
4. **Forecaster supervision pipeline.** Decide precompute-offline vs on-the-fly for the
   full-clip ground-truth future positions/latents (disk vs compute, mirrors companion spec task 3).
5. **`F` token shape into memory/condition.** Confirm D4RT `F` flattens to `[B, N, C]` compatibly
   with `DualMemoryBank` dims (add a projection if `C` ≠ memory dim).
6. **Pick the checkpoint + sanity-check reproduction quality (R5).** Default to
   `OpenD4RT_48CLIP_9Mix_NoCropAUG`. Before the probe, run the repo's own WorldTrack eval on a
   small set to confirm the weights load and produce sane 3D tracks — establishes the encoder
   isn't broken independent of the forecasting question.
