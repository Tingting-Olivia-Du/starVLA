# Geo-MemoryVLA Phase-C: image_window dataloader — Design Spec

**Date:** 2026-06-29
**Framework host:** starVLA (`/workspace/tingting/starVLA`)
**Status:** Design approved, pending implementation plan
**Parent:** [Geo-MemoryVLA design](2026-06-29-geo-memoryvla-design.md) §7 (Phase-C follow-up)

---

## 0. Coding convention (unchanged from parent)

Every new file / modified block carries the `# [Geo-MemoryVLA]` marker + a pointer to this
spec. Lifted/derived behavior names its source. See parent spec §0.

---

## 1. Goal

Provide each LIBERO sample with a multi-frame `image_window` (past context + future chunk) so
Geo-MemoryVLA's imagination (the vendored `VGGTWorldModel`) can actually train. This unlocks the
`imagination.enabled` switch that the parent project shipped defaulted **off** because the
single-frame dataloader made the vendored world model crash / degenerate.

The window must satisfy the vendored model's requirement `F >= context_size + chunk_size`
(`vggt_world/model.py:_stage1_forward` raises `ValueError` otherwise).

---

## 2. Approved decisions (locked)

| Decision | Choice |
| --- | --- |
| Window frame range | **Past context + future chunk**: `[-(context_size-1) .. 0 .. +chunk_size]`, aligning with the vendored model's `slice_frames(context, context+chunk)` (teacher-forced: past conditions, future supervises). |
| Generation mechanism | **Dedicated `image_window` modality** (own `ModalityConfig` + `delta_indices`), NOT a change to the shared `observation_indices`. Keeps the main observation path = current frame only. |
| Window-length source of truth | **Derived** from `framework.imagination.{context_size, horizon}` — never hand-filled twice. `image_window_indices = list(range(-(context_size-1), chunk_size+1))` where `chunk_size = horizon`. |
| `enable_image_window` default | **`true`**, but generation **follows `imagination.enabled`**: window is produced only when imagination is on (imagination off → not declared → zero extra frame reads). Default config also flips `imagination.enabled: true` so it trains out of the box. |
| Frame format | `image_window` = `List[F][num_views]` of PIL.Image 224×224, element-wise identical to `image`. Matches `_build_image_window`'s existing stack into `[B, F*views, 3, H, W]`. |

---

## 3. Architecture & data flow

```
data_config (LIBERO DataConfig)
  image_window_indices = list(range(-(context_size-1), chunk_size+1))   # e.g. ctx=2,chunk=4 -> [-1,0,1,2,3,4]
  (declared only when imagination active)
        │
        ▼
ModalityConfig(delta_indices=image_window_indices, modality_keys=<video keys>)
  └─ reuses get_video's existing episode-boundary clamp (np.min/np.max) + first_last padding
        │
        ▼
__getitem__ -> _pack_sample
  ├─ sample["image"]        = observation current frame (1 frame/cam, UNCHANGED)
  └─ sample["image_window"] = all window frames -> List[F][views] of PIL 224x224
        │   (produced only when the image_window modality is declared)
        ▼
GeoMemoryVLA._build_image_window(examples)
  └─ EXISTING: reads e["image_window"] -> [B, F*views, 3, H, W]; now F>1 for real
        │
        ▼
ImaginationAdapter -> vendored VGGTWorldModel  (F >= context+chunk satisfied; no crash/degeneration)
        └─ training_loss (stage1/stage2 flow-forcing) + imagine_tokens (subgoal)
```

**Reuse, don't reinvent:** frame fetching, episode-boundary clamping
(`get_video`: `np.maximum(idx,0)` / `np.minimum(idx, traj_len-1)`), and `first_last` padding are
all existing machinery. We only declare the indices and pack one extra key.

---

## 4. Components

### 4.1 data_config — dedicated `image_window` modality (NEW ~15 LOC)
- File: `starVLA/dataloader/gr00t_lerobot/data_config.py` (the LIBERO DataConfig class).
- Add `image_window_indices` derived from imagination's `context_size`/`horizon`.
- Add a `ModalityConfig` entry for `image_window` (modality_keys = the same video keys as
  observation; `delta_indices = image_window_indices`).
- Gating (explicit): the `image_window` modality is declared **iff**
  `enable_image_window AND imagination.enabled`. `enable_image_window` defaults true, so in practice
  the window follows `imagination.enabled`. Setting `enable_image_window=false` is an escape hatch to
  force it off even with imagination on (e.g. debugging). When not declared → zero extra frame reads.
- `observation_indices` stays `[0]` — the main VLA observation path is untouched.

### 4.2 datasets.py `_pack_sample` — emit `image_window` (NEW ~12 LOC)
- File: `starVLA/dataloader/gr00t_lerobot/datasets.py`, `_pack_sample`.
- When the `image_window` modality data is present, pack its `F` frames into
  `sample["image_window"] = List[F][num_views]` of PIL 224×224 (same `Image.fromarray(...).resize((224,224))`
  path as `image`). The current frame for `sample["image"]` still comes from the observation modality.
- When absent (imagination off), do not add the key — `_build_image_window` then degenerates as today.

### 4.3 GeoMemoryVLA — error-handling hardening (MODIFY ~15 LOC)
- File: `starVLA/model/framework/VLM4A/GeoMemoryVLA.py`.
- **Construct-time check** (`__init__`): when `imagination.enabled=true`, assert the derived window
  length `>= context_size + chunk_size`; else `raise ValueError` pointing to this spec.
- **Runtime upgrade** (`_build_image_window`): when `imagination.enabled=true` but a batch example
  lacks `"image_window"`, **upgrade the existing warn-once to a hard `raise`** (silent single-frame
  degeneration while believing imagination trains is the dangerous failure). Imagination-off path
  is unchanged (single-frame fallback fine).
- **Frame-count check**: after building the window tensor, assert `F == len(image_window_indices)`
  to catch padding/indexing surprises.

### 4.4 Config (MODIFY ~3 LOC)
- File: `examples/LIBERO/train_files/geo_memoryvla_libero.yaml`.
- Add `datasets.vla_data.enable_image_window: true`.
- Flip `framework.imagination.enabled` back to `true` (prerequisite now satisfied).

---

## 5. Boundary & error handling

- **Episode boundaries (reused):** windows crossing an episode head/tail are clamped by
  `get_video`'s existing `np.maximum/np.minimum`, and out-of-range frames are `first_last`-padded
  (repeat first/last frame). No cross-episode bleed; no new code. Semantically the world model sees
  a "still" transition at boundaries — acceptable.
- **Three error-handling lines of defense** (§4.3): construct-time window-length assert; runtime
  hard-raise on missing window when imagination is on; post-build frame-count assert.

---

## 6. Testing

| Test | Verifies | Type |
| --- | --- | --- |
| `test_image_window_indices_derivation` | `range(-(ctx-1), chunk+1)` correct; length == ctx+chunk | fast |
| `test_pack_sample_emits_window` | given fake multi-frame data, `_pack_sample` emits `image_window` = List[F][views] PIL 224×224, and `image` is still the current frame | fast (mock, no real data) |
| `test_episode_boundary_clamp` | base_index=0 (episode head): past frames clamp to frame 0, no bleed into prior episode | fast |
| `test_build_image_window_raises_without_window_when_imag_on` | imagination on + no `image_window` → raises (defense line 2) | fast |
| `test_smoke_geo_only_imagination` (extends Task 8) | real LIBERO data + geo_only + imagination on: forward runs, `imagination_loss` finite, F satisfies the world model | GPU slow (GPUs 0-3) |

---

## 7. Known constraints / risks

1. **GPU smoke stays geo_only.** Full dual+imagination needs Qwen3-VL, and this env's transformers
   4.53.2 lacks `Qwen3VLForConditionalGeneration`. The smoke runs `geo_only + imagination on` —
   this DOES exercise the genuinely new chain (window → VGGTWorldModel → imagination_loss), just not
   the semantic stream. Dual+imagination verification waits on transformers ≥ 4.54.
2. **Frame-read cost when imagination on.** Each sample reads `F` frames/cam instead of 1. Bounded by
   `context_size + chunk_size` (default 6). Acceptable; gated off when imagination is off.
3. **Window length is config-derived.** If a user overrides `context_size`/`horizon` inconsistently,
   the construct-time assert (§4.3) fails fast rather than crashing deep in the vendored model.

---

## 8. Code-change magnitude

| Type | Files | Est. LOC |
| --- | --- | --- |
| NEW modality declaration | `data_config.py` | ~15 |
| NEW packing | `datasets.py` `_pack_sample` | ~12 |
| MODIFY hardening | `GeoMemoryVLA.py` (construct/runtime/frame-count guards) | ~15 |
| MODIFY config | `geo_memoryvla_libero.yaml` | ~3 |
| NEW tests | `tests/geomemvla/` | ~120 |

**Rating:** small-to-medium. The heavy lifting (frame indexing, clamping, padding) is reused from
the existing `delta_indices` machinery; new code is the modality declaration, the pack step, the
three guards, and tests.
