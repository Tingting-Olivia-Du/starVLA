# PointWorld Distillation — V0 Baseline + V1 Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.
>
> **GATE:** This plan MUST NOT begin until the PW-PROBE gate (plan `2026-07-01-lang-pointworld-probe-gate.md`, Task 8) records **PASS**. Training on gate-rejected teacher labels is a spec §4 kill-criterion violation. If the gate fails after both checkpoints + LIBERO fine-tune, STOP and do not execute this plan.

**Goal:** Build the RGB-only language-conditioned starVLA student with an optional PointWorld-distillation stream, and demonstrate the controlled experiment `framework.pw_distill.enabled ∈ {off,on}` — V0 (off, behavior cloning) vs V1 (on, + cached teacher point-flow supervision) — on LIBERO closed-loop success rate.

**Architecture:** Reuse the `GeoMemoryVLA` host verbatim (Qwen-VL semantic stream + GR00T flow-matching action head + LIBERO dataloader + eval). Add three things gated by one config flag: (C3) an offline teacher-flow cache built by running the frozen PointWorld teacher over demo actions; (C4) a student point-flow head that regresses cached teacher flow and whose tokens condition the action head; (C5) a `pw_distill_loss` term added beside the existing `imagination_loss` at [GeoMemoryVLA.py:345-348](../../../starVLA/model/framework/VLM4A/GeoMemoryVLA.py). The teacher never runs at test — the student produces every action.

**Tech Stack:** Python 3.10, PyTorch, starVLA (`GeoMemoryVLA`, `get_action_model`, `ConditionAssembler`), PointWorld teacher wrapper (from the probe-gate plan), the frozen depth model chosen in Phase-0 (sim depth / Depth-Anything-3 / VGGT), LIBERO datasets, the starVLA env at `/workspace/ghsun/miniconda3/envs/starVLA`.

**Spec:** [`docs/superpowers/specs/2026-07-01-lang-pointworld-distill-design.md`](../specs/2026-07-01-lang-pointworld-distill-design.md) §3.1 (C3/C4/C5), §5 (student), §6 (V0/V1), §9 (eval).

## Global Constraints

- **Marker** `[LangPointWorld]`; PointWorld provenance line per spec §0.
- **Controlled experiment:** `off` and `on` runs are byte-identical except `framework.pw_distill.enabled`. Same seed, same data order, same steps. Any success-rate delta must be attributable to distillation alone.
- **Memory OFF in V1** (spec §5.1): `world_state`/dual-memory disabled; only the semantic stream + pw_distill stream active. Keeps the experiment single-variable.
- **Coordinate frame:** teacher cache, GT, and student flow all in the current-timestep **EE-centric frame** (spec §4), via `starVLA/model/modules/langpw/ee_frame.py` from the probe-gate plan. Same-frame assert enforced.
- **Test time = student only.** No PointWorld, no cache, no teacher-flow-target loaded during LIBERO closed-loop eval. Verify by asserting the eval path never imports `pointworld_teacher`.
- **GPUs 2,3** (`CUDA_VISIBLE_DEVICES=2,3`); big-disk routing for checkpoints/cache/TMPDIR/wandb (`/workspace/...`, not `/tmp`); `HF_HOME=/root/angli/hf_cache`.
- **Sparse points N=256** (spec §5.3): 96 target-object · 48 target-region · 48 gripper/contact · 64 background/context, masks from LIBERO sim segmentation.
- **Flow horizon** `Hf∈{4..8}`, action horizon = host default, `K=2` window.
- **Loss:** `L = L_action + λ_flow · L_flow`, `L_flow = Σ w_i · smoothL1(F_student − F_teacher)`, no other terms in V1.

---

## File Structure

| Path | Responsibility |
|---|---|
| `starVLA/model/modules/langpw/point_sampling.py` | Hybrid sparse-point selection (§5.3) from LIBERO sim segmentation + teacher flow magnitude → per-sample point indices + weights `w_i` + role labels. |
| `starVLA/model/modules/langpw/teacher_cache.py` | Cache reader/writer: `(episode_id, timestep) → {teacher_flow_ee[N,Hf,3], point_role[N], weight[N], visibility[N]}`. On-disk `.npz` under a cache root; the training loop only imports THIS, never the teacher. |
| `tools/pw_build_teacher_cache.py` | Offline driver: for every training sample, run `PointWorldTeacher.imagine` on the demo action chunk, sample points, transform flow to EE frame, write cache. Where PointWorld runs — once, offline, GPU 2,3. |
| `starVLA/model/modules/langpw/point_flow_head.py` | (C4) `cond → F_student[B,N,Hf,3]` (+ visibility); a small cross-attention head over `cond`. Produces flow tokens for the action-head condition. |
| `starVLA/model/modules/langpw/pw_distill_loss.py` | (C5) weighted smooth-L1 between `F_student` and cached `F_teacher` in EE frame, with same-frame assert. |
| `starVLA/model/framework/VLM4A/GeoMemoryVLA.py` | MODIFY (~40 LOC, `[LangPointWorld]`): build flow head + loss when `pw_distill.enabled`; append flow tokens to streams pre-assembly; add `pw_distill_loss` to the returned dict. |
| `starVLA/model/framework/VLM4A/config` (the framework config dataclass) | MODIFY: add `pw_distill` config block (`enabled, cache_root, n_points, flow_horizon, loss_scale, feed_flow_tokens`). |
| dataloader (episode-fields path) | MODIFY: attach cached teacher flow to each example by `(episode_id, timestep)` when `pw_distill.enabled`. |
| configs: `v0_baseline.yaml`, `v1_distill.yaml` | Two run configs differing only by `pw_distill.enabled`. |
| `tests/geomemvla/test_pw_point_sampling.py`, `test_pw_flow_head.py`, `test_pw_distill_loss.py`, `test_pw_distill_config_loads.py`, `test_pw_distill_smoke_gpu.py` | Unit + config + smoke tests, matching the geomemvla test suite conventions. |

---

## Task 1: Hybrid point sampling (§5.3)

**Files:** Create `starVLA/model/modules/langpw/point_sampling.py`; Test `tests/geomemvla/test_pw_point_sampling.py`.

**Interfaces:**
- Consumes: teacher `scene_coord0[Ns,3]`, teacher `scene_flow[Hf,Ns,3]` (for motion magnitude), LIBERO seg masks (`object_mask, target_mask, gripper_mask` as `[Ns] bool`, derived in the cache driver from sim segmentation).
- Produces: `sample_points(scene_coord0, scene_flow, object_mask, target_mask, gripper_mask, n_points=256, alloc=(96,48,48,64)) -> dict{indices[N], weight[N], role[N]}` where role ∈ {0:obj,1:target,2:gripper,3:bg} and weight up-weights obj/gripper/moving, down-weights bg/static.

- [ ] **Step 1: failing test**

```python
# tests/geomemvla/test_pw_point_sampling.py
# [LangPointWorld] Hybrid sparse point selection (spec §5.3), CPU.
import torch
from starVLA.model.modules.langpw.point_sampling import sample_points

def test_allocation_and_weights():
    Ns = 2000; Hf = 4
    coord = torch.rand(Ns, 3); flow = torch.rand(Hf, Ns, 3) * 0.01
    obj = torch.zeros(Ns, bool); obj[:500] = True
    tgt = torch.zeros(Ns, bool); tgt[500:900] = True
    grip = torch.zeros(Ns, bool); grip[900:1100] = True
    out = sample_points(coord, flow, obj, tgt, grip, n_points=256, alloc=(96,48,48,64))
    assert out["indices"].shape == (256,)
    assert out["role"].shape == (256,) and out["weight"].shape == (256,)
    # object + gripper points must carry higher mean weight than background
    w = out["weight"]; role = out["role"]
    assert w[(role==0)].mean() > w[(role==3)].mean()

def test_handles_missing_masks_gracefully():
    Ns=500; out = sample_points(torch.rand(Ns,3), torch.rand(2,Ns,3),
        torch.zeros(Ns,bool), torch.zeros(Ns,bool), torch.zeros(Ns,bool), n_points=128, alloc=(32,32,32,32))
    assert out["indices"].shape == (128,)  # falls back to motion/uniform when masks empty
```

- [ ] **Step 2: run, expect FAIL** — `cd /workspace/tingting/starVLA && /workspace/ghsun/miniconda3/envs/starVLA/bin/python -m pytest tests/geomemvla/test_pw_point_sampling.py -v`

- [ ] **Step 3: implement** `point_sampling.py` — allocate per role from each mask's indices (fill shortfall from top motion-magnitude points, then uniform); weights: obj=1.0, gripper=1.0, target=0.7, bg=0.3, multiplied by a motion factor `(1+||flow||)`; return indices/weight/role tensors. Header carries the marker + spec pointer.

- [ ] **Step 4: run, expect PASS**

- [ ] **Step 5: commit** — `git add -f ...point_sampling.py tests/geomemvla/test_pw_point_sampling.py && git commit -m "[LangPointWorld] Hybrid sparse point sampling (C3 §5.3) + tests"`

---

## Task 2: Teacher-flow cache reader/writer + offline build driver (C3)

**Files:** Create `starVLA/model/modules/langpw/teacher_cache.py`, `tools/pw_build_teacher_cache.py`; Test `tests/geomemvla/test_pw_teacher_cache.py`.

**Interfaces:**
- Consumes: `PointWorldTeacher.imagine` (probe-gate plan Task 5), `build_pw_sample` (probe-gate Task 3), `flow_to_ee_frame` (probe-gate Task 2), `sample_points` (Task 1), the Phase-0 depth provider, LIBERO sim segmentation.
- Produces:
  - `write_cache(cache_root, episode_id, timestep, payload: dict) -> None` and `read_cache(cache_root, episode_id, timestep) -> dict|None` with keys `teacher_flow_ee[N,Hf,3], point_role[N], weight[N], visibility[N]`.
  - CLI `pw_build_teacher_cache.py --libero-glob ... --cache-root ... --depth {sim,depth_anything3,vggt} --horizon Hf` iterating all demos×timesteps.

- [ ] **Step 1: failing test** (round-trip on tmp dir)

```python
# tests/geomemvla/test_pw_teacher_cache.py
# [LangPointWorld] Cache round-trip (CPU, no model).
import numpy as np
from starVLA.model.modules.langpw.teacher_cache import write_cache, read_cache

def test_roundtrip(tmp_path):
    payload = {"teacher_flow_ee": np.zeros((256,4,3),np.float32),
               "point_role": np.zeros(256,np.int64),
               "weight": np.ones(256,np.float32),
               "visibility": np.ones((256,4),np.float32)}
    write_cache(str(tmp_path), "ep1", 3, payload)
    got = read_cache(str(tmp_path), "ep1", 3)
    assert got is not None and got["teacher_flow_ee"].shape == (256,4,3)
    assert read_cache(str(tmp_path), "ep1", 99) is None  # missing -> None
```

- [ ] **Step 2: run, expect FAIL**

- [ ] **Step 3: implement** `teacher_cache.py` (npz keyed `{episode_id}__{timestep}.npz` under cache_root) and `pw_build_teacher_cache.py`. The driver:
  1. builds the depth provider per `--depth` (sim: replay LIBERO env with `camera_depths=True`; else load Depth-Anything-3 or VGGT frozen and predict per-frame — **engineering choice: prefer sim depth if Phase-0 found it available (metric-accurate); else Depth-Anything-3 for its metric-depth head**);
  2. per (demo, t): `build_pw_sample` → `teacher.imagine(demo_action_chunk)` → derive seg masks → `sample_points` → `flow_to_ee_frame` on the selected points' flow → `write_cache`.
  Runs on GPU 2,3, `torch.no_grad()`.

- [ ] **Step 4: run cache round-trip test, expect PASS**; then a **real small cache build** (`--episodes 4`) to prove the driver runs end-to-end on GPU.

- [ ] **Step 5: commit**

---

## Task 3: Student point-flow head (C4)

**Files:** Create `starVLA/model/modules/langpw/point_flow_head.py`; Test `tests/geomemvla/test_pw_flow_head.py`.

**Interfaces:**
- Consumes: host `cond[B,L,H]` (assembler output dim).
- Produces: `class PointFlowHead(nn.Module)`:
  - `__init__(self, cond_dim, n_points=256, flow_horizon=8, hidden=512)`.
  - `forward(self, cond, encoder_attention_mask=None) -> dict{"flow": Tensor[B,N,Hf,3], "flow_tokens": Tensor[B,N,hidden]}` — N learned point queries cross-attend `cond`; an MLP maps each query to `Hf×3` flow; `flow_tokens` are the pre-projection query states for the action-head condition.

- [ ] **Step 1: failing test** (shape + grad flow)

```python
# tests/geomemvla/test_pw_flow_head.py
# [LangPointWorld] Point-flow head shapes + gradients (CPU).
import torch
from starVLA.model.modules.langpw.point_flow_head import PointFlowHead

def test_shapes_and_grad():
    B,L,H = 2, 10, 256
    head = PointFlowHead(cond_dim=H, n_points=64, flow_horizon=4, hidden=128)
    cond = torch.randn(B,L,H, requires_grad=True)
    out = head(cond)
    assert out["flow"].shape == (B,64,4,3)
    assert out["flow_tokens"].shape == (B,64,128)
    out["flow"].sum().backward()
    assert cond.grad is not None
```

- [ ] **Step 2: run, expect FAIL**

- [ ] **Step 3: implement** a `nn.MultiheadAttention`-based head: `nn.Parameter` query bank `[N,hidden]`, cross-attend to a linear-projected `cond`, LayerNorm+MLP → `[B,N,Hf*3]`. Marker header.

- [ ] **Step 4: run, expect PASS**

- [ ] **Step 5: commit**

---

## Task 4: Distillation loss (C5)

**Files:** Create `starVLA/model/modules/langpw/pw_distill_loss.py`; Test `tests/geomemvla/test_pw_distill_loss.py`.

**Interfaces:**
- Consumes: `ee_frame.assert_same_frame`.
- Produces: `pw_distill_loss(flow_student[B,N,Hf,3], flow_teacher[B,N,Hf,3], weight[B,N], frame_id_student, frame_id_teacher) -> Tensor[]` — weighted smooth-L1, same-frame asserted.

- [ ] **Step 1: failing test**

```python
# tests/geomemvla/test_pw_distill_loss.py
import torch, pytest
from starVLA.model.modules.langpw.pw_distill_loss import pw_distill_loss

def test_zero_when_equal():
    f = torch.randn(2,8,4,3); w = torch.ones(2,8)
    assert float(pw_distill_loss(f, f.clone(), w, 0, 0)) < 1e-6

def test_weight_scales_and_frame_guard():
    fs = torch.zeros(1,2,1,3); ft = torch.ones(1,2,1,3); w = torch.tensor([[2.0,0.0]])
    loss = pw_distill_loss(fs, ft, w, 3, 3)
    assert loss > 0
    with pytest.raises(ValueError):
        pw_distill_loss(fs, ft, w, 3, 4)   # frame mismatch
```

- [ ] **Step 2-5:** run→FAIL, implement (`F.smooth_l1_loss(reduction='none')` × `weight`, mean over valid), run→PASS, commit.

---

## Task 5: Config block + dataloader wiring

**Files:** MODIFY the framework config dataclass (add `pw_distill`), MODIFY the episode-fields dataloader path to attach cached flow; Create `configs/v0_baseline.yaml`, `configs/v1_distill.yaml`; Test `tests/geomemvla/test_pw_distill_config_loads.py`.

**Interfaces:**
- Produces: `framework.pw_distill = {enabled:bool, cache_root:str, n_points:256, flow_horizon:8, loss_scale:1.0, feed_flow_tokens:true}`. Dataloader adds `example["pw_teacher"] = read_cache(...)` when enabled (None-safe).

- [ ] **Step 1: failing config test** (mirror `test_d4rt_config_loads.py`): load `v1_distill.yaml`, assert `config.framework.pw_distill.enabled is True` and `v0_baseline.yaml` False.
- [ ] **Step 2: run, expect FAIL**
- [ ] **Step 3: implement** config dataclass field + two YAMLs (identical but for the flag + memory off) + dataloader attach (guarded, None when cache miss).
- [ ] **Step 4: run, expect PASS**
- [ ] **Step 5: commit**

---

## Task 6: Wire flow head + loss into GeoMemoryVLA (the ~40-LOC seam)

**Files:** MODIFY `starVLA/model/framework/VLM4A/GeoMemoryVLA.py`; Test `tests/geomemvla/test_pw_distill_smoke_gpu.py`.

**Interfaces:**
- Consumes: `PointFlowHead` (Task 3), `pw_distill_loss` (Task 4), `example["pw_teacher"]` (Task 5).
- Produces: `forward()` returns `{"action_loss":..., "pw_distill_loss": λ·L_flow}` when enabled; flow tokens appended to the stream list before `_assemble` when `feed_flow_tokens`.

- [ ] **Step 1: failing smoke test** — build `GeoMemoryVLA` from `v1_distill.yaml` with tiny dims, one fake batch carrying a `pw_teacher` payload, assert `out["pw_distill_loss"]` present, finite, and `>0`; and that a `v0_baseline` build's `out` has NO `pw_distill_loss` key.

- [ ] **Step 2: run, expect FAIL**

- [ ] **Step 3: implement** in `__init__`: if `fw.pw_distill.enabled` → `self.pw_flow_head = PointFlowHead(cond_dim, ...)`, `self.pw_loss_scale = fw.pw_distill.loss_scale`. In `forward()` (beside the `imagination_loss` block at line ~345):
```python
# [LangPointWorld] point-flow distillation stream (spec §3.1 C4/C5); gated, memory-off in V1.
pw_loss = None
if getattr(self, "use_pw_distill", False):
    flow_out = self.pw_flow_head(cond, encoder_attention_mask=mask)  # cond pre-repeat
    teach = _stack_pw_teacher(examples, device)                      # [B,N,Hf,3], weight, frame_id
    if self.pw_feed_flow_tokens:
        streams = streams + [flow_out["flow_tokens"]]  # appended before _assemble upstream
    pw_loss = self.pw_loss_scale * pw_distill_loss(
        flow_out["flow"], teach["flow"], teach["weight"], teach["frame_id"], teach["frame_id"])
...
if pw_loss is not None:
    out["pw_distill_loss"] = pw_loss
```
(Note: if `feed_flow_tokens`, the flow head must run BEFORE `_assemble`; restructure so streams include flow tokens — this is the one non-trivial ordering change. Keep it greppable.)

- [ ] **Step 4: run smoke on GPU 2,3, expect PASS**

- [ ] **Step 5: commit**

---

## Task 7: Verify test-time student-only (no teacher leak)

**Files:** Test `tests/geomemvla/test_pw_no_teacher_at_eval.py`.

- [ ] **Step 1:** test that importing/constructing the deployment/eval path with `v1_distill.yaml` never imports `pointworld_teacher` and that a forward with no `pw_teacher` in the batch still yields a valid action (flow head runs, loss is simply not computed). Assert `sys.modules` has no `...pointworld_teacher` after an eval-mode forward.
- [ ] **Step 2-5:** run→FAIL if leak, fix (lazy-import teacher only in the cache tool), run→PASS, commit. This is the guarantee behind "test time = student only" (spec §1.2).

---

## Task 8: Build full teacher cache, then train V0 and V1

**Files:** run scripts `tools/train_pw_v0.sh`, `tools/train_pw_v1.sh`.

- [ ] **Step 1: Build the full teacher cache** for the target LIBERO suite(s) using the Phase-0 depth source. GPU 2,3, background. Verify cache count == Σ demos×usable-timesteps.
- [ ] **Step 2: Train V0** (`v0_baseline.yaml`, `pw_distill.enabled=false`, memory off) on GPU 2,3 with a fixed seed. Record final checkpoint + LIBERO eval success rate.
- [ ] **Step 3: Train V1** (`v1_distill.yaml`, identical but `enabled=true`) — same seed, same steps, same data order.
- [ ] **Step 4: Eval both** on LIBERO-Spatial/Object/Goal/Long closed loop; record success rates. **The V1−V0 delta is the result.**
- [ ] **Step 5: Ablation** — V1 with `feed_flow_tokens=false` (aux-loss-only) as the third bar (spec §9).
- [ ] **Step 6:** write results + eval commands to a results note; commit.

---

## V2 / V3 sketches (NOT implemented in this plan — spec §6)

- **V2 (language-grounded desired flow):** add object/target/motion language queries to the flow head; grounding loss `L_ground = CE(point-attention, seg-mask)`. Point-flow becomes language-conditioned, not just teacher-reconstruction.
- **V3 (= original "Plan A", online differentiable inverse consistency):** student action → **differentiable** FK → frozen PointWorld → `L_inverse = ||F_rollout − F_desired||`. This is where differentiable FK through urdfpy and per-sample PointWorld backprop are tackled — only after V1 proves value. Budget for the ~1B-PTv3 backprop cost then.

---

## Self-Review

- Spec §3.1 C3 (offline cache) → Tasks 1,2 ✓; C4 (flow head) → Task 3 ✓; C5 (loss+wiring) → Tasks 4,5,6 ✓.
- Spec §1.2 (test = student only) → Task 7 ✓. Spec §5.1 (memory off) → Global Constraints + Task 5 configs ✓. Spec §5.3 (hybrid N=256) → Task 1 ✓. Spec §4 (EE frame) → reused from probe-gate plan, asserted in Tasks 2,4 ✓.
- Spec §6 ladder V0/V1 → Task 8; V2/V3 → sketches only (correctly deferred) ✓. Spec §9 eval + ablation → Task 8 steps 4,5 ✓.
- Gate dependency stated in header ✓. Type consistency: `sample_points`→`{indices,weight,role}` used in Task 2; `PointFlowHead.forward`→`{flow,flow_tokens}` used in Task 6; `pw_distill_loss(...)` signature consistent Tasks 4,6; `read_cache/write_cache` consistent Tasks 2,5. ✓
- Known non-trivial point flagged honestly: Task 6 stream-ordering when `feed_flow_tokens` (flow head must precede `_assemble`) — called out, not hidden.
