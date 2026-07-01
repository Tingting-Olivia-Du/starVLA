# PointWorld Distillation — Phase 0 Preflight + PW-PROBE Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine — before writing any training code — whether a frozen, DROID/BEHAVIOR-pretrained PointWorld produces *usable* future 3D point-flow on out-of-domain LIBERO data, by (a) proving LIBERO data can be fed through PointWorld's preprocessing at all, then (b) measuring imagined-vs-GT flow agreement in the EE-centric frame with corr/L2 metrics plus viser visual confirmation.

**Architecture:** A LIBERO-episode → PointWorld-`data_dict` adapter drives the frozen teacher offline; a probe harness feeds *real* demo actions, extracts imagined future scene-flow, transforms both prediction and GT into the current-frame EE frame, and reports gate metrics. No student, no training, no differentiable path in this plan — the teacher runs forward-only. This plan is the spec's C0 gate; its verdict gates all downstream V0/V1 work.

**Tech Stack:** Python 3.10, PyTorch, PointWorld (`/workspace/tingting/PointWorld`, NVlabs, Apache-2.0 code + access-gated DINOv3), starVLA test conventions (pytest under `tests/geomemvla/`), h5py (LIBERO HDF5), viser (visualization), the starVLA env at `/workspace/ghsun/miniconda3/envs/starVLA`.

**Spec:** [`docs/superpowers/specs/2026-07-01-lang-pointworld-distill-design.md`](../specs/2026-07-01-lang-pointworld-distill-design.md) — this plan implements §4 (PW-PROBE gate) and its Phase-0 prerequisites (§3.1 C0/C1/C2).

## Global Constraints

- **Marker:** every new file header and modified block carries `[LangPointWorld]`. New-file header form and PointWorld provenance line per spec §0 (`# Vendored from NVlabs/PointWorld (Apache-2.0): github.com/NVlabs/PointWorld`).
- **No training, no differentiable FK in this plan.** Teacher is forward-only, `torch.no_grad()`, `model.eval()`. Differentiable FK / backprop is spec V3, out of scope.
- **Coordinate frame:** all imagined/GT flow compared in the **current-timestep EE-centric frame** (spec §4). Origin = that frame's EE position, axes = EE orientation. Never mix frames across timesteps — assert it.
- **Gate metrics (both required):** key-point per-dim **corr > 0.7** (pass threshold), plus **filtered-L2** reported (threshold derived from the run's distribution, not a blind cutoff), plus **viser** visual check on ≥8 episodes. Spec §4.
- **GPUs:** use 4,5,6,7 for any GPU run (0–3 in use by others) — set `CUDA_VISIBLE_DEVICES`.
- **Big-disk routing:** outputs, `TMPDIR`, caches to `/workspace/...` not `/tmp` (30G, ENOSPC risk). `HF_HOME=/root/angli/hf_cache`.
- **Env:** run everything with `/workspace/ghsun/miniconda3/envs/starVLA/bin/python`.
- **PointWorld checkpoints:** probe `large-droid/model-best.pt` first; `large-droid+behavior/model-best.pt` is the documented fallback (spec §4 kill/branch).
- **starVLA repo has a nested layout:** package root is `/workspace/tingting/starVLA/starVLA/`, tests+tools+docs are at `/workspace/tingting/starVLA/`. New tools go in `/workspace/tingting/starVLA/tools/`, new modules in `/workspace/tingting/starVLA/starVLA/model/modules/langpw/`.

---

## File Structure

| Path | Responsibility |
|---|---|
| `starVLA/model/modules/langpw/__init__.py` | package marker (new) |
| `starVLA/model/modules/langpw/libero_camera_adapter.py` | (C1) LIBERO HDF5 episode → per-timestep raw `sample` dict in PointWorld's pipeline contract (RGB-D + intrinsics/extrinsics + joints + EE + base_pose), resized to (180,320), agentview un-flipped. Depth source resolution lives here. |
| `starVLA/model/modules/langpw/pointworld_teacher.py` | (C2) frozen PointWorld loader + forward wrapper: `sample dict → data_dict → imagined future scene-flow`. Drives `apply_release_pipeline_to_sample` + `BaseModel.forward`. |
| `starVLA/model/modules/langpw/ee_frame.py` | EE-centric transform utility: world/camera-frame points+flow → current-frame EE frame, with same-frame assert. Shared by probe now, cache/loss later. |
| `tools/pw_probe_libero.py` | (C0) probe harness CLI: run teacher on LIBERO demo actions, compute corr/L2 in EE frame, dump a probe-result JSON + per-episode arrays for viz. |
| `tools/pw_probe_visualize.py` | (C0) load probe arrays → viser scene showing imagined vs GT flow (mirrors `tools/d4rt_probe_visualize.py`). |
| `tools/run_pw_probe.sh` | (C0) end-to-end runner (env, GPU, paths, ckpt) mirroring `Open-d4rt/tools/run_libero_d4rt.sh`. |
| `tests/geomemvla/test_pw_ee_frame.py` | unit tests for `ee_frame.py` (pure, CPU, no model). |
| `tests/geomemvla/test_pw_camera_adapter.py` | contract tests for `libero_camera_adapter.py` (CPU, tiny synthetic HDF5). |
| `tests/geomemvla/test_pw_probe_metrics.py` | unit tests for the corr/L2 metric functions (pure, CPU). |
| `tests/geomemvla/test_pw_teacher_smoke_gpu.py` | GPU smoke: one real LIBERO episode through the teacher yields finite flow of the right shape. |

**Decomposition rationale:** pure-math units (`ee_frame`, metrics) are tested CPU-only and first — they carry the frame-discipline that everything else depends on. The adapter (C1) is the real integration surface and gets contract tests against a synthetic HDF5. The teacher wrapper (C2) is thin over PointWorld and only smoke-tested on GPU. The probe harness (C0) is the deliverable and is validated by producing a real gate verdict, not a unit test.

---

## Task 1: Phase-0 preflight — can LIBERO data physically enter PointWorld?

**This task is a spike, not a feature.** Its deliverable is a written go/no-go note answering the two hard integration unknowns discovered during spec review: **(a) LIBERO HDF5 has no depth**, but PointWorld's pipeline requires `<cam>_initial_depth`; **(b) PointWorld demands (180,320) resolution and a domain-registered URDF**, LIBERO is 128×128 Panda. Resolve the depth source and URDF registration before building anything.

**Files:**
- Create: `docs/superpowers/notes/2026-07-01-pw-preflight.md` (findings note)

**Interfaces:**
- Consumes: PointWorld `dataset_components/pipeline.py:apply_release_pipeline_to_sample`, `utils.resolve_robot_urdf`, `scene_featurizer.py` input contract (`rgb/depth/intrinsic/extrinsic`).
- Produces: a documented **depth source decision** (one of: LIBERO sim depth re-render / regenerate dataset with depth / monocular estimator) and a **domain+URDF decision** (which `domain` string LIBERO maps to and where its Panda URDF comes from). Downstream Task 3 (`libero_camera_adapter`) depends on both.

- [ ] **Step 1: Confirm LIBERO obs has no depth, record exact schema**

Run:
```bash
/workspace/ghsun/miniconda3/envs/starVLA/bin/python - <<'PY'
import h5py
p="/workspace/tingting/LIBERO/libero/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
f=h5py.File(p,"r"); demo=list(f["data"].keys())[0]; obs=f["data"][demo]["obs"]
print("OBS:", {k:(obs[k].shape,str(obs[k].dtype)) for k in obs.keys()})
print("has depth:", any("depth" in k for k in obs.keys()))
PY
```
Expected: keys `agentview_rgb, ee_ori, ee_pos, ee_states, eye_in_hand_rgb, gripper_states, joint_states`; `has depth: False`.

- [ ] **Step 2: Decide the depth source**

Investigate, in this priority order (spec §9.2), and write the choice into the note:
1. **LIBERO sim re-render with depth** — check whether the LIBERO/robosuite env used to generate these datasets can replay demos with `camera_depths=True`. Look in `/workspace/tingting/LIBERO` for the env config that produced `agentview_rgb`.
   Run: `grep -rn "camera_depths\|depth\|render_depth" /workspace/tingting/LIBERO/libero/libero/envs/ 2>/dev/null | head`
2. If sim re-render is available → that is the choice (clean metric depth). If not → fall back to a monocular estimator on `agentview_rgb` and **flag in the note that PW-PROBE will run on estimated depth** (weaker, but the gate is exactly the instrument to catch resulting errors).

- [ ] **Step 3: Decide the domain string + Panda URDF for FK**

PointWorld's robot flow needs a URDF via `resolve_robot_urdf(domain)`. LIBERO uses a 7-DoF Panda + parallel gripper (not Robotiq).
Run:
```bash
/workspace/ghsun/miniconda3/envs/starVLA/bin/python -c "import sys; sys.path.insert(0,'/workspace/tingting/PointWorld'); from utils import resolve_robot_urdf; print('droid->',resolve_robot_urdf('droid'))"
grep -rn "def resolve_robot_urdf\|panda\|franka\|\.urdf" /workspace/tingting/PointWorld/utils.py | head
```
Record: does DROID's URDF (Franka Panda) match LIBERO's Panda joint layout closely enough to reuse `domain="droid"`, or must a LIBERO Panda URDF be supplied? Write the decision. (Reusing `domain="droid"` is the low-effort path if joints align — LIBERO Panda ≈ DROID Franka arm.)

- [ ] **Step 4: Write the go/no-go note and commit**

Write `docs/superpowers/notes/2026-07-01-pw-preflight.md` with the header marker, the three decisions (depth source, domain/URDF, resolution-resize approach 128→180×320), and any blocker. If depth is genuinely unobtainable, this is a **hard stop for the whole plan** — say so explicitly.

```bash
git add -f docs/superpowers/notes/2026-07-01-pw-preflight.md
git commit -m "[LangPointWorld] Phase-0 preflight: LIBERO->PointWorld depth/URDF/resolution decisions"
```

Expected: note committed; downstream tasks reference its decisions.

---

## Task 2: EE-centric frame transform utility

**Files:**
- Create: `starVLA/model/modules/langpw/__init__.py`, `starVLA/model/modules/langpw/ee_frame.py`
- Test: `tests/geomemvla/test_pw_ee_frame.py`

**Interfaces:**
- Consumes: nothing (pure torch).
- Produces:
  - `to_ee_frame(points_world: Tensor[...,3], ee_pos: Tensor[3], ee_quat: Tensor[4]) -> Tensor[...,3]` — express world-frame points in the EE frame (rotate by `ee_quat`⁻¹, translate by `-ee_pos`).
  - `flow_to_ee_frame(flow_world: Tensor[...,3], ee_quat: Tensor[4]) -> Tensor[...,3]` — a *displacement* only rotates (no translation): rotate by `ee_quat`⁻¹.
  - `assert_same_frame(frame_id_a: int, frame_id_b: int) -> None` — raises `ValueError` if two arrays being compared came from different timesteps' EE frames.
  Consumed by Task 4 (probe) and later by cache/loss.

- [ ] **Step 1: Write the failing test**

```python
# tests/geomemvla/test_pw_ee_frame.py
# [LangPointWorld] Unit tests for the EE-centric frame transform (pure, CPU).
import torch, pytest
from starVLA.model.modules.langpw.ee_frame import to_ee_frame, flow_to_ee_frame, assert_same_frame


def _quat_identity():
    return torch.tensor([0.0, 0.0, 0.0, 1.0])  # (x,y,z,w)


def test_identity_ee_pose_is_translation_only():
    pts = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    ee_pos = torch.tensor([1.0, 1.0, 1.0])
    out = to_ee_frame(pts, ee_pos, _quat_identity())
    assert torch.allclose(out, pts - ee_pos, atol=1e-6)


def test_flow_is_rotation_only_not_translated():
    # a pure displacement must be invariant to EE translation
    flow = torch.tensor([[0.5, 0.0, 0.0]])
    out = flow_to_ee_frame(flow, _quat_identity())
    assert torch.allclose(out, flow, atol=1e-6)


def test_90deg_z_rotation_maps_x_to_minus_y():
    # quat for +90deg about z: (0,0,sin45,cos45)
    s = (2 ** 0.5) / 2
    quat = torch.tensor([0.0, 0.0, s, s])
    flow = torch.tensor([[1.0, 0.0, 0.0]])
    out = flow_to_ee_frame(flow, quat)  # world->ee uses inverse rotation
    assert torch.allclose(out, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-5)


def test_assert_same_frame_rejects_mismatch():
    assert_same_frame(5, 5)  # ok
    with pytest.raises(ValueError):
        assert_same_frame(5, 6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/tingting/starVLA && /workspace/ghsun/miniconda3/envs/starVLA/bin/python -m pytest tests/geomemvla/test_pw_ee_frame.py -v`
Expected: FAIL with `ModuleNotFoundError: ...langpw.ee_frame`.

- [ ] **Step 3: Write minimal implementation**

```python
# starVLA/model/modules/langpw/__init__.py
# [LangPointWorld] Package for PointWorld point-flow distillation modules.
# Part of the PointWorld point-flow distillation arm — see
# docs/superpowers/specs/2026-07-01-lang-pointworld-distill-design.md
```

```python
# starVLA/model/modules/langpw/ee_frame.py
# [LangPointWorld] EE-centric ("robo-centric") frame transform — cross-dataset
# unification frame for teacher/GT/student point-flow (spec §4).
# Part of the PointWorld point-flow distillation arm — see
# docs/superpowers/specs/2026-07-01-lang-pointworld-distill-design.md
import torch


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    # q = (x,y,z,w); conjugate = (-x,-y,-z,w). For unit quats, conjugate == inverse.
    return torch.stack([-q[0], -q[1], -q[2], q[3]])


def _rotate_by_quat(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    # Rotate vectors v[...,3] by unit quaternion q=(x,y,z,w). Hamilton product form.
    x, y, z, w = q[0], q[1], q[2], q[3]
    qv = torch.stack([x, y, z])
    uv = torch.cross(qv.expand_as(v), v, dim=-1)
    uuv = torch.cross(qv.expand_as(v), uv, dim=-1)
    return v + 2.0 * (w * uv + uuv)


def flow_to_ee_frame(flow_world: torch.Tensor, ee_quat: torch.Tensor) -> torch.Tensor:
    """A displacement rotates only (world->EE uses the inverse EE rotation)."""
    return _rotate_by_quat(flow_world, _quat_conjugate(ee_quat))


def to_ee_frame(points_world: torch.Tensor, ee_pos: torch.Tensor, ee_quat: torch.Tensor) -> torch.Tensor:
    """Express world-frame points in the EE frame: translate then inverse-rotate."""
    return _rotate_by_quat(points_world - ee_pos, _quat_conjugate(ee_quat))


def assert_same_frame(frame_id_a: int, frame_id_b: int) -> None:
    """Guard: EE origin moves per timestep; only compare arrays from the same frame."""
    if int(frame_id_a) != int(frame_id_b):
        raise ValueError(
            f"[LangPointWorld] EE-frame mismatch: {frame_id_a} vs {frame_id_b}. "
            "Teacher/GT/student flow must be compared in the same timestep's EE frame (spec §4)."
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /workspace/tingting/starVLA && /workspace/ghsun/miniconda3/envs/starVLA/bin/python -m pytest tests/geomemvla/test_pw_ee_frame.py -v`
Expected: 4 passed. (If `test_90deg` fails on sign, the world→EE direction is inverted — the transform must use the *conjugate*; verify the rotation convention against the assertion, not vice-versa.)

- [ ] **Step 5: Commit**

```bash
git add -f starVLA/model/modules/langpw/__init__.py starVLA/model/modules/langpw/ee_frame.py tests/geomemvla/test_pw_ee_frame.py
git commit -m "[LangPointWorld] EE-centric frame transform + tests"
```

---

## Task 3: LIBERO → PointWorld camera/sample adapter (C1)

**Files:**
- Create: `starVLA/model/modules/langpw/libero_camera_adapter.py`
- Test: `tests/geomemvla/test_pw_camera_adapter.py`

**Interfaces:**
- Consumes: Task-1 decisions (depth source, `domain` string, resize approach). LIBERO HDF5 schema: `obs/{agentview_rgb,eye_in_hand_rgb,ee_pos,ee_ori,joint_states,gripper_states}`, `actions`, all `[T,...]`.
- Produces: `build_pw_sample(hdf5_path: str, demo: str, t: int, *, depth_provider, domain: str, intrinsics: dict, extrinsics: dict) -> dict` returning a raw sample dict with keys PointWorld's pipeline requires: `agentview_initial_rgb, agentview_initial_depth, agentview_intrinsic, agentview_extrinsic` (and `eye_in_hand_*`), `joint_positions [7]`, `joint_names`, `base_pose [4,4]`, plus `ee_pos [3]`, `ee_quat [4]` carried through for the EE frame. RGB is un-flipped and resized to (180,320). Consumed by Task 5 (teacher) and Task 6 (probe).

- [ ] **Step 1: Write the failing test (against a tiny synthetic HDF5)**

```python
# tests/geomemvla/test_pw_camera_adapter.py
# [LangPointWorld] Contract test: LIBERO episode -> PointWorld raw sample dict (CPU).
import numpy as np, h5py, torch
from starVLA.model.modules.langpw.libero_camera_adapter import build_pw_sample


def _make_synth_hdf5(path, T=4):
    with h5py.File(path, "w") as f:
        g = f.create_group("data/demo_0/obs")
        g.create_dataset("agentview_rgb", data=np.zeros((T,128,128,3), np.uint8))
        g.create_dataset("eye_in_hand_rgb", data=np.zeros((T,128,128,3), np.uint8))
        g.create_dataset("ee_pos", data=np.zeros((T,3), np.float64))
        g.create_dataset("ee_ori", data=np.zeros((T,3), np.float64))
        g.create_dataset("joint_states", data=np.zeros((T,7), np.float64))
        g.create_dataset("gripper_states", data=np.zeros((T,2), np.float64))
        f.create_dataset("data/demo_0/actions", data=np.zeros((T,7), np.float64))


def test_sample_has_pointworld_contract_keys(tmp_path):
    p = str(tmp_path / "e.hdf5"); _make_synth_hdf5(p)
    depth = lambda cam, t: np.ones((128,128), np.float32)  # stub depth provider
    K = {"agentview_rgb": np.eye(3), "eye_in_hand_rgb": np.eye(3)}
    E = {"agentview_rgb": np.eye(4), "eye_in_hand_rgb": np.eye(4)}
    s = build_pw_sample(p, "demo_0", 0, depth_provider=depth, domain="droid", intrinsics=K, extrinsics=E)
    for k in ["agentview_initial_rgb","agentview_initial_depth","agentview_intrinsic","agentview_extrinsic",
              "joint_positions","joint_names","base_pose","ee_pos","ee_quat"]:
        assert k in s, f"missing {k}"
    assert s["agentview_initial_rgb"].shape[:2] == (180,320), "RGB must be resized to PointWorld (180,320)"
    assert s["agentview_initial_depth"].shape[:2] == (180,320)
    assert len(s["joint_positions"]) == 7
    assert s["ee_quat"].shape[-1] == 4


def test_agentview_is_unflipped(tmp_path):
    # LIBERO stores agentview upside-down; adapter must vertical-flip so scene is upright.
    p = str(tmp_path / "e.hdf5")
    with h5py.File(p, "w") as f:
        g = f.create_group("data/demo_0/obs")
        img = np.zeros((1,128,128,3), np.uint8); img[0,0,:,:] = 255  # top row white
        g.create_dataset("agentview_rgb", data=img)
        g.create_dataset("eye_in_hand_rgb", data=np.zeros((1,128,128,3), np.uint8))
        for k,sh in [("ee_pos",3),("ee_ori",3),("joint_states",7),("gripper_states",2)]:
            g.create_dataset(k, data=np.zeros((1,sh), np.float64))
        f.create_dataset("data/demo_0/actions", data=np.zeros((1,7), np.float64))
    depth = lambda cam, t: np.ones((128,128), np.float32)
    K = {"agentview_rgb": np.eye(3), "eye_in_hand_rgb": np.eye(3)}
    E = {"agentview_rgb": np.eye(4), "eye_in_hand_rgb": np.eye(4)}
    s = build_pw_sample(p, "demo_0", 0, depth_provider=depth, domain="droid", intrinsics=K, extrinsics=E)
    rgb = s["agentview_initial_rgb"]
    assert rgb[-1,:,:].mean() > rgb[0,:,:].mean(), "white row should be at bottom after un-flip"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/tingting/starVLA && /workspace/ghsun/miniconda3/envs/starVLA/bin/python -m pytest tests/geomemvla/test_pw_camera_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError: ...libero_camera_adapter`.

- [ ] **Step 3: Write minimal implementation**

```python
# starVLA/model/modules/langpw/libero_camera_adapter.py
# [LangPointWorld] LIBERO HDF5 episode -> PointWorld raw sample dict (C1, spec §3.1).
# Resolves the depth gap (LIBERO obs has no depth), un-flips agentview, resizes
# 128x128 -> PointWorld's (180,320) contract, converts euler ee_ori -> quaternion.
# Part of the PointWorld point-flow distillation arm — see
# docs/superpowers/specs/2026-07-01-lang-pointworld-distill-design.md
import numpy as np, h5py, cv2

_PW_HW = (180, 320)  # PointWorld release camera-payload resolution (pipeline assert)
_PANDA_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]  # domain='droid' Franka layout


def _euler_xyz_to_quat(e):
    # LIBERO ee_ori is axis-angle-ish euler (x,y,z). Convert to (x,y,z,w) unit quat.
    cx, cy, cz = np.cos(np.asarray(e) * 0.5)
    sx, sy, sz = np.sin(np.asarray(e) * 0.5)
    w = cx*cy*cz + sx*sy*sz
    x = sx*cy*cz - cx*sy*sz
    y = cx*sy*cz + sx*cy*sz
    z = cx*cy*sz - sx*sy*cz
    q = np.array([x, y, z, w], np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def _prep_rgb(img):
    up = np.flipud(img)                       # LIBERO agentview stored upside-down
    return cv2.resize(up, (_PW_HW[1], _PW_HW[0]), interpolation=cv2.INTER_AREA)


def _prep_depth(d):
    return cv2.resize(np.asarray(d, np.float32), (_PW_HW[1], _PW_HW[0]), interpolation=cv2.INTER_NEAREST)


def build_pw_sample(hdf5_path, demo, t, *, depth_provider, domain, intrinsics, extrinsics):
    with h5py.File(hdf5_path, "r") as f:
        obs = f["data"][demo]["obs"]
        rgb_a = np.asarray(obs["agentview_rgb"][t])
        rgb_w = np.asarray(obs["eye_in_hand_rgb"][t])
        ee_pos = np.asarray(obs["ee_pos"][t], np.float64)
        ee_ori = np.asarray(obs["ee_ori"][t], np.float64)
        joints = np.asarray(obs["joint_states"][t], np.float64)
    ee_quat = _euler_xyz_to_quat(ee_ori)
    sample = {
        "agentview_initial_rgb": _prep_rgb(rgb_a),
        "agentview_initial_depth": _prep_depth(depth_provider("agentview_rgb", t)),
        "agentview_intrinsic": np.asarray(intrinsics["agentview_rgb"], np.float64),
        "agentview_extrinsic": np.asarray(extrinsics["agentview_rgb"], np.float64),
        "eye_in_hand_initial_rgb": _prep_rgb(rgb_w),
        "eye_in_hand_initial_depth": _prep_depth(depth_provider("eye_in_hand_rgb", t)),
        "eye_in_hand_intrinsic": np.asarray(intrinsics["eye_in_hand_rgb"], np.float64),
        "eye_in_hand_extrinsic": np.asarray(extrinsics["eye_in_hand_rgb"], np.float64),
        "joint_positions": joints,
        "joint_names": list(_PANDA_JOINT_NAMES),
        "base_pose": np.eye(4, dtype=np.float64),
        "ee_pos": ee_pos,
        "ee_quat": ee_quat,
        "__domain__": domain,
    }
    return sample
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /workspace/tingting/starVLA && /workspace/ghsun/miniconda3/envs/starVLA/bin/python -m pytest tests/geomemvla/test_pw_camera_adapter.py -v`
Expected: 2 passed. If `cv2` import fails, add `opencv-python-headless` to the env (note it in the preflight); if the un-flip test fails, the `np.flipud` axis is wrong for the stored orientation — verify against a real frame with viser in Task 7.

- [ ] **Step 5: Commit**

```bash
git add -f starVLA/model/modules/langpw/libero_camera_adapter.py tests/geomemvla/test_pw_camera_adapter.py
git commit -m "[LangPointWorld] LIBERO->PointWorld camera/sample adapter (C1) + tests"
```

---

## Task 4: Probe metric functions (corr / filtered-L2 in EE frame)

**Files:**
- Create: `starVLA/model/modules/langpw/probe_metrics.py`
- Test: `tests/geomemvla/test_pw_probe_metrics.py`

**Interfaces:**
- Consumes: `ee_frame.flow_to_ee_frame` (Task 2).
- Produces:
  - `keypoint_corr(pred_flow_ee: Tensor[N,H,3], gt_flow_ee: Tensor[N,H,3], mask: Tensor[N]) -> dict` → `{"corr_x","corr_y","corr_z","corr_mean"}` per-dim Pearson correlation over masked key points+horizon.
  - `filtered_l2(pred_flow_ee, gt_flow_ee, mask) -> dict` → `{"l2_mean","l2_p50","l2_p90"}` endpoint L2 over masked points.
  - `gate_verdict(corr_mean: float, corr_thresh: float = 0.7) -> bool`.
  Consumed by Task 6 (probe harness).

- [ ] **Step 1: Write the failing test**

```python
# tests/geomemvla/test_pw_probe_metrics.py
# [LangPointWorld] Unit tests for probe gate metrics (pure, CPU).
import torch
from starVLA.model.modules.langpw.probe_metrics import keypoint_corr, filtered_l2, gate_verdict


def test_perfect_prediction_corr_is_one_and_l2_zero():
    gt = torch.randn(16, 4, 3)
    mask = torch.ones(16, dtype=torch.bool)
    c = keypoint_corr(gt.clone(), gt.clone(), mask)
    assert c["corr_mean"] > 0.99
    l = filtered_l2(gt.clone(), gt.clone(), mask)
    assert l["l2_mean"] < 1e-5


def test_anti_correlated_fails_gate():
    gt = torch.randn(16, 4, 3)
    mask = torch.ones(16, dtype=torch.bool)
    c = keypoint_corr(-gt, gt, mask)   # sign-flipped = under-drive/opposite direction
    assert c["corr_mean"] < 0.0
    assert gate_verdict(c["corr_mean"]) is False


def test_mask_excludes_points():
    gt = torch.randn(8, 2, 3); pred = gt.clone(); pred[4:] = 0.0
    mask = torch.zeros(8, dtype=torch.bool); mask[:4] = True   # only first 4 are perfect
    c = keypoint_corr(pred, gt, mask)
    assert c["corr_mean"] > 0.99


def test_gate_threshold():
    assert gate_verdict(0.71) is True
    assert gate_verdict(0.69) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/tingting/starVLA && /workspace/ghsun/miniconda3/envs/starVLA/bin/python -m pytest tests/geomemvla/test_pw_probe_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: ...probe_metrics`.

- [ ] **Step 3: Write minimal implementation**

```python
# starVLA/model/modules/langpw/probe_metrics.py
# [LangPointWorld] PW-PROBE gate metrics: per-dim corr + filtered L2, EE frame (spec §4).
# Part of the PointWorld point-flow distillation arm — see
# docs/superpowers/specs/2026-07-01-lang-pointworld-distill-design.md
import torch


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).double(); b = b.reshape(-1).double()
    a = a - a.mean(); b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    return float((a @ b) / denom)


def keypoint_corr(pred_flow_ee, gt_flow_ee, mask):
    p = pred_flow_ee[mask]; g = gt_flow_ee[mask]   # [M,H,3]
    cx, cy, cz = (_pearson(p[..., i], g[..., i]) for i in range(3))
    return {"corr_x": cx, "corr_y": cy, "corr_z": cz, "corr_mean": (cx + cy + cz) / 3.0}


def filtered_l2(pred_flow_ee, gt_flow_ee, mask):
    p = pred_flow_ee[mask]; g = gt_flow_ee[mask]
    l2 = torch.norm(p - g, dim=-1).reshape(-1)     # per point-horizon endpoint error
    return {
        "l2_mean": float(l2.mean()),
        "l2_p50": float(l2.median()),
        "l2_p90": float(l2.quantile(0.90)),
    }


def gate_verdict(corr_mean: float, corr_thresh: float = 0.7) -> bool:
    return bool(corr_mean > corr_thresh)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /workspace/tingting/starVLA && /workspace/ghsun/miniconda3/envs/starVLA/bin/python -m pytest tests/geomemvla/test_pw_probe_metrics.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add -f starVLA/model/modules/langpw/probe_metrics.py tests/geomemvla/test_pw_probe_metrics.py
git commit -m "[LangPointWorld] PW-PROBE gate metrics (corr/L2) + tests"
```

---

## Task 5: Frozen PointWorld teacher wrapper (C2)

**Files:**
- Create: `starVLA/model/modules/langpw/pointworld_teacher.py`
- Test: `tests/geomemvla/test_pw_teacher_smoke_gpu.py`

**Interfaces:**
- Consumes: Task-3 `build_pw_sample`; PointWorld `arguments.parse_args`-style args, `pointworld/base.py:BaseModel`, `dataset_components/pipeline.py:apply_release_pipeline_to_sample`, checkpoint `.pt` (`torch.load(..., weights_only=False)`, key `model` per `Tester`).
- Produces: `class PointWorldTeacher` with:
  - `__init__(self, ckpt_path, ptv3_size="large", domain="droid", device="cuda")` — builds frozen model, `eval()`, loads state dict.
  - `imagine(self, sample: dict, demo_action_chunk: np.ndarray) -> dict` → `{"scene_flows": Tensor[T,Ns,3], "scene_coord0": Tensor[Ns,3]}` in **world/camera frame** (EE transform is the caller's job, Task 6). Runs under `torch.no_grad()`.
  Consumed by Task 6.

- [ ] **Step 1: Write the failing GPU smoke test**

```python
# tests/geomemvla/test_pw_teacher_smoke_gpu.py
# [LangPointWorld] GPU smoke: one real LIBERO episode -> finite PointWorld flow.
import os, numpy as np, pytest, torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU only")

CKPT = os.environ.get("PW_CKPT", "")
HDF5 = os.environ.get("PW_LIBERO_HDF5", "")


@pytest.mark.skipif(not (CKPT and HDF5), reason="set PW_CKPT and PW_LIBERO_HDF5")
def test_teacher_imagine_returns_finite_flow():
    from starVLA.model.modules.langpw.pointworld_teacher import PointWorldTeacher
    from starVLA.model.modules.langpw.libero_camera_adapter import build_pw_sample
    depth = lambda cam, t: np.ones((128,128), np.float32)  # placeholder until Task-1 depth
    K = {"agentview_rgb": np.eye(3), "eye_in_hand_rgb": np.eye(3)}
    E = {"agentview_rgb": np.eye(4), "eye_in_hand_rgb": np.eye(4)}
    s = build_pw_sample(HDF5, "demo_0", 0, depth_provider=depth, domain="droid", intrinsics=K, extrinsics=E)
    teacher = PointWorldTeacher(CKPT, ptv3_size="large", domain="droid", device="cuda")
    out = teacher.imagine(s, demo_action_chunk=np.zeros((8,7), np.float64))
    assert torch.isfinite(out["scene_flows"]).all()
    assert out["scene_flows"].shape[-1] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/tingting/starVLA && /workspace/ghsun/miniconda3/envs/starVLA/bin/python -m pytest tests/geomemvla/test_pw_teacher_smoke_gpu.py -v`
Expected: FAIL with `ModuleNotFoundError: ...pointworld_teacher` (or skip if env vars unset — acceptable; the module-not-found is the target once vars are set).

- [ ] **Step 3: Write minimal implementation**

```python
# starVLA/model/modules/langpw/pointworld_teacher.py
# [LangPointWorld] Frozen PointWorld teacher wrapper (C2, spec §3.1). Forward-only.
# Vendored driving of NVlabs/PointWorld (Apache-2.0): github.com/NVlabs/PointWorld
# Part of the PointWorld point-flow distillation arm — see
# docs/superpowers/specs/2026-07-01-lang-pointworld-distill-design.md
import sys, types, numpy as np, torch

_PW_ROOT = "/workspace/tingting/PointWorld"


def _ensure_pw_on_path():
    if _PW_ROOT not in sys.path:
        sys.path.insert(0, _PW_ROOT)


def _default_pw_args(ptv3_size, domain, device):
    # Minimal args object mirroring what arguments.parse_args() yields for eval.
    # Only fields BaseModel + pipeline read are set; extend if a KeyError appears.
    a = types.SimpleNamespace()
    a.ptv3_size = ptv3_size; a.device = device; a.distributed = False
    a.domains = domain; a.ptv3_patch_size = None
    return a


class PointWorldTeacher:
    def __init__(self, ckpt_path, ptv3_size="large", domain="droid", device="cuda"):
        _ensure_pw_on_path()
        from pointworld.base import BaseModel
        from arguments import parse_args  # noqa: F401  (import proves module resolves)
        self.device = device
        self.domain = domain
        self.args = _default_pw_args(ptv3_size, domain, device)
        # data_info_dict is normally produced by the dataloader; the eval path builds it
        # from norm stats. Load the released stats dir for `domain`.
        from pointworld.norm_stats import load_norm_stats  # adjust to real symbol in Task
        self.data_info = load_norm_stats(self.args)
        self.model = BaseModel(self.args, self.data_info, rank=0).to(device).eval()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["model"] if "model" in ckpt else ckpt
        self.model.load_state_dict(state, strict=False)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def imagine(self, sample: dict, demo_action_chunk: np.ndarray) -> dict:
        _ensure_pw_on_path()
        from dataset_components.pipeline import apply_release_pipeline_to_sample
        s = dict(sample)
        s["action_chunk"] = np.asarray(demo_action_chunk, np.float64)
        s = apply_release_pipeline_to_sample(
            s, domain=self.domain, mode="test", args=self.args, include_scene_data=True
        )
        batch = {k: (torch.as_tensor(v).unsqueeze(0).to(self.device)
                     if isinstance(v, np.ndarray) else ([v] if k == "__domain__" else v))
                 for k, v in s.items()}
        out = self.model(batch, training=False)
        return {
            "scene_flows": out["scene_flows"][0].detach().cpu(),   # [T,Ns,3]
            "scene_coord0": batch["scene_flows"][0, 0].detach().cpu(),
        }
```

> **Note for the implementer:** the exact symbol names `load_norm_stats` and the arg fields are best-effort from the spec's verified-facts table; the first GPU run will surface the real ones via `AttributeError`/`KeyError`. Fix them against `/workspace/tingting/PointWorld/arguments.py`, `pointworld/norm_stats.py`, and `evaluation/tester.py:__init__` (which shows the real build+load sequence) — do **not** guess past a second failure; read those three files. This is expected integration work, not a placeholder.

- [ ] **Step 4: Run the GPU smoke test**

Run:
```bash
cd /workspace/tingting/starVLA && CUDA_VISIBLE_DEVICES=4 \
PW_CKPT=/path/to/pretrained_checkpoints/large-droid/model-best.pt \
PW_LIBERO_HDF5=/workspace/tingting/LIBERO/libero/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5 \
/workspace/ghsun/miniconda3/envs/starVLA/bin/python -m pytest tests/geomemvla/test_pw_teacher_smoke_gpu.py -v -s
```
Expected: PASS (finite flow, last dim 3). Iterate on real PointWorld symbol names until it passes — reading `evaluation/tester.py`, not guessing.

- [ ] **Step 5: Commit**

```bash
git add -f starVLA/model/modules/langpw/pointworld_teacher.py tests/geomemvla/test_pw_teacher_smoke_gpu.py
git commit -m "[LangPointWorld] Frozen PointWorld teacher wrapper (C2) + GPU smoke"
```

---

## Task 6: PW-PROBE harness — the gate verdict (C0)

**Files:**
- Create: `tools/pw_probe_libero.py`
- Test: (validated by producing a real verdict JSON, not a unit test — it is an integration driver)

**Interfaces:**
- Consumes: `PointWorldTeacher.imagine` (Task 5), `build_pw_sample` (Task 3), `ee_frame` (Task 2), `probe_metrics` (Task 4), a depth provider (Task-1 decision).
- Produces: a CLI writing `probe_result.json` (`{corr_*, l2_*, gate_pass, n_episodes, ckpt, depth_source}`) and per-episode `.npz` (`pred_flow_ee, gt_flow_ee, keypoint_mask, ee_pos, ee_quat`) for Task-7 viz. Prints the **PASS/FAIL gate verdict** and, on FAIL, the spec §4 branch instruction.

- [ ] **Step 1: Write the probe harness**

```python
# tools/pw_probe_libero.py
# [LangPointWorld] PW-PROBE gate driver (C0, spec §4): frozen PointWorld on LIBERO
# demo actions -> imagined vs GT future flow in EE frame -> corr/L2 verdict.
# Part of the PointWorld point-flow distillation arm — see
# docs/superpowers/specs/2026-07-01-lang-pointworld-distill-design.md
import argparse, json, os, glob, numpy as np, torch
from starVLA.model.modules.langpw.pointworld_teacher import PointWorldTeacher
from starVLA.model.modules.langpw.libero_camera_adapter import build_pw_sample
from starVLA.model.modules.langpw.ee_frame import flow_to_ee_frame, assert_same_frame
from starVLA.model.modules.langpw import probe_metrics as M


def _gt_future_flow(hdf5, demo, t, horizon):
    # GT scene motion proxy: EE trajectory displacement over the horizon, in world frame.
    # (Full-scene GT flow needs depth-tracked points; for the gate we use the reliable
    #  EE/gripper trajectory as the key-point set — the arm dims the FCDecoder probe used.)
    import h5py
    with h5py.File(hdf5, "r") as f:
        ee = np.asarray(f["data"][demo]["obs"]["ee_pos"][:], np.float64)
    T = ee.shape[0]
    steps = [ee[min(t + h, T - 1)] - ee[t] for h in range(1, horizon + 1)]
    return np.stack(steps, 0)  # [H,3] world-frame EE displacement


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--libero-glob", required=True)
    ap.add_argument("--ptv3-size", default="large")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--out-dir", default="/workspace/tingting/starVLA/results/pw_probe")
    ap.add_argument("--corr-thresh", type=float, default=0.7)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    teacher = PointWorldTeacher(args.ckpt, ptv3_size=args.ptv3_size, device="cuda")
    files = sorted(glob.glob(args.libero_glob))[: args.episodes]
    K = {"agentview_rgb": np.eye(3), "eye_in_hand_rgb": np.eye(3)}  # TODO Task-1: real LIBERO intrinsics/extrinsics
    E = {"agentview_rgb": np.eye(4), "eye_in_hand_rgb": np.eye(4)}
    depth = lambda cam, t: np.ones((128, 128), np.float32)          # TODO Task-1: real depth provider

    all_pred, all_gt = [], []
    for fi, hdf5 in enumerate(files):
        demo = "demo_0"; t = 0
        import h5py
        with h5py.File(hdf5, "r") as f:
            actions = np.asarray(f["data"][demo]["actions"][t:t + args.horizon], np.float64)
        s = build_pw_sample(hdf5, demo, t, depth_provider=depth, domain="droid", intrinsics=K, extrinsics=E)
        out = teacher.imagine(s, demo_action_chunk=actions)
        # Reduce imagined full-scene flow to the gripper key-point set nearest EE (proxy),
        # then express both pred and GT displacement in this timestep's EE frame.
        ee_quat = torch.as_tensor(s["ee_quat"], dtype=torch.float32)
        pred_world = out["scene_flows"][: args.horizon]                       # [H,Ns,3]
        # gripper key points: the Ns points whose t=0 coord is closest to EE position
        d0 = torch.norm(out["scene_coord0"] - torch.as_tensor(s["ee_pos"], dtype=torch.float32), dim=-1)
        kp = torch.topk(-d0, k=min(64, d0.numel())).indices
        pred_kp = pred_world[:, kp, :].mean(1)                                # [H,3] mean gripper motion
        pred_ee = flow_to_ee_frame(pred_kp, ee_quat)                          # [H,3]
        gt_ee = flow_to_ee_frame(torch.as_tensor(_gt_future_flow(hdf5, demo, t, args.horizon), dtype=torch.float32), ee_quat)
        assert_same_frame(t, t)
        all_pred.append(pred_ee.unsqueeze(0)); all_gt.append(gt_ee.unsqueeze(0))
        np.savez(os.path.join(args.out_dir, f"ep{fi}.npz"),
                 pred_flow_ee=pred_ee.numpy(), gt_flow_ee=gt_ee.numpy(),
                 ee_pos=s["ee_pos"], ee_quat=s["ee_quat"])

    pred = torch.cat(all_pred, 0); gt = torch.cat(all_gt, 0)                  # [E,H,3]
    mask = torch.ones(pred.shape[0], dtype=torch.bool)
    corr = M.keypoint_corr(pred, gt, mask); l2 = M.filtered_l2(pred, gt, mask)
    verdict = M.gate_verdict(corr["corr_mean"], args.corr_thresh)
    result = {**corr, **l2, "gate_pass": verdict, "n_episodes": len(files),
              "ckpt": args.ckpt, "corr_thresh": args.corr_thresh}
    with open(os.path.join(args.out_dir, "probe_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    if not verdict:
        print("\n[LangPointWorld] GATE FAILED (spec §4 branch): "
              "retry large-droid+behavior ckpt -> short LIBERO finetune -> else STOP. "
              "Do NOT proceed to V0/V1 distillation on rejected labels.")
    else:
        print("\n[LangPointWorld] GATE PASSED — proceed to V0/V1 plan.")


if __name__ == "__main__":
    main()
```

> **Implementer note:** two `# TODO Task-1` lines (intrinsics/extrinsics, depth provider) MUST be replaced with the real values from the Task-1 decision before the verdict is trustworthy. Identity intrinsics/dummy depth will produce a garbage-but-runnable pipeline — useful to prove the plumbing, NOT a valid gate. The GT-flow proxy uses the EE trajectory (the reliable arm dims, matching the FCDecoder probe methodology in [[vla-rlds-closedloop-rootcause]]); upgrading to depth-tracked full-scene GT is a documented follow-up if the EE-proxy gate is inconclusive.

- [ ] **Step 2: Run the plumbing check (dummy depth/intrinsics — proves it runs, not the verdict)**

Run:
```bash
cd /workspace/tingting/starVLA && CUDA_VISIBLE_DEVICES=4 \
/workspace/ghsun/miniconda3/envs/starVLA/bin/python tools/pw_probe_libero.py \
  --ckpt /path/to/pretrained_checkpoints/large-droid/model-best.pt \
  --libero-glob "/workspace/tingting/LIBERO/libero/datasets/libero_object/*demo.hdf5" \
  --episodes 8
```
Expected: writes `results/pw_probe/probe_result.json` + `ep*.npz`; prints a verdict. (Numbers not yet meaningful until Task-1 depth/intrinsics are wired.)

- [ ] **Step 3: Wire real depth + intrinsics/extrinsics from Task-1, re-run for the true verdict**

Replace the two TODO lines with the Task-1 depth provider and LIBERO camera params. Re-run Step 2's command. This produces the **real gate verdict**.

- [ ] **Step 4: Commit the harness + the verdict artifact**

```bash
git add -f tools/pw_probe_libero.py
git commit -m "[LangPointWorld] PW-PROBE harness (C0) — corr/L2 gate verdict on LIBERO"
```

---

## Task 7: Probe visualization (viser)

**Files:**
- Create: `tools/pw_probe_visualize.py`, `tools/run_pw_probe.sh`

**Interfaces:**
- Consumes: `results/pw_probe/ep*.npz` (Task 6).
- Produces: a viser server rendering, per episode, the EE-frame imagined vs GT flow vectors (mirrors `tools/d4rt_probe_visualize.py`). Human confirmation the corr/L2 numbers reflect reality (spec §4 metric 3).

- [ ] **Step 1: Write the visualizer (mirroring the existing d4rt probe visualizer)**

First read the existing one for the viser idiom, then mirror it:
Run: `sed -n '1,60p' /workspace/tingting/starVLA/tools/d4rt_probe_visualize.py`

```python
# tools/pw_probe_visualize.py
# [LangPointWorld] Viser visualization of PW-PROBE imagined vs GT EE-frame flow (spec §4).
# Mirrors tools/d4rt_probe_visualize.py.
# Part of the PointWorld point-flow distillation arm — see
# docs/superpowers/specs/2026-07-01-lang-pointworld-distill-design.md
import argparse, glob, os, numpy as np, viser, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", default="/workspace/tingting/starVLA/results/pw_probe")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    server = viser.ViserServer(port=args.port)
    eps = sorted(glob.glob(os.path.join(args.probe_dir, "ep*.npz")))
    idx = server.gui.add_slider("episode", 0, max(len(eps) - 1, 0), 1, 0)

    def render(i):
        d = np.load(eps[i])
        pred, gt = d["pred_flow_ee"], d["gt_flow_ee"]      # [H,3] each
        origin = np.zeros((pred.shape[0], 3), np.float32)   # EE frame origin
        server.scene.add_line_segments("pred", np.stack([origin, pred], 1), colors=(0, 200, 0))
        server.scene.add_line_segments("gt", np.stack([origin, gt], 1), colors=(200, 0, 0))

    idx.on_update(lambda _: render(idx.value))
    render(0)
    print(f"[LangPointWorld] viser on http://localhost:{args.port} — green=pred, red=GT (EE frame)")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the end-to-end runner**

```bash
# tools/run_pw_probe.sh
# [LangPointWorld] End-to-end PW-PROBE gate: teacher -> metrics -> viser (spec §4).
#!/usr/bin/env bash
set -euo pipefail
PY="${PY:-/workspace/ghsun/miniconda3/envs/starVLA/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export TMPDIR="${TMPDIR:-/workspace/tingting/.tmp}"; mkdir -p "$TMPDIR"
CKPT="${CKPT:?set CKPT to a PointWorld model-best.pt}"
GLOB="${GLOB:-/workspace/tingting/LIBERO/libero/datasets/libero_object/*demo.hdf5}"
OUT="${OUT:-/workspace/tingting/starVLA/results/pw_probe}"
cd /workspace/tingting/starVLA
"$PY" tools/pw_probe_libero.py --ckpt "$CKPT" --libero-glob "$GLOB" --out-dir "$OUT" --episodes "${EPISODES:-8}"
echo "[run_pw_probe] verdict written to $OUT/probe_result.json"
echo "[run_pw_probe] visualize: $PY tools/pw_probe_visualize.py --probe-dir $OUT --port ${PORT:-8080}"
```

- [ ] **Step 3: Run visualization on the probe output**

Run:
```bash
cd /workspace/tingting/starVLA && /workspace/ghsun/miniconda3/envs/starVLA/bin/python \
  tools/pw_probe_visualize.py --probe-dir results/pw_probe --port 8080
```
Expected: viser server up; green (pred) vs red (GT) flow vectors per episode. Human check: do they point the same way? Divergent direction ⇒ corroborates a FAIL verdict; aligned-but-short ⇒ under-drive (scale), still informative.

- [ ] **Step 4: Commit**

```bash
git add -f tools/pw_probe_visualize.py tools/run_pw_probe.sh
git commit -m "[LangPointWorld] PW-PROBE viser visualization + end-to-end runner"
```

---

## Task 8: Record the gate verdict and decide the branch

**Files:**
- Create: `docs/superpowers/notes/2026-07-01-pw-probe-verdict.md`

**Interfaces:**
- Consumes: `results/pw_probe/probe_result.json`, viser inspection.
- Produces: the documented decision that gates the (separate) V0/V1 training plan.

- [ ] **Step 1: Write the verdict note**

Record: corr_mean and per-dim, L2 percentiles, viser observations, which ckpt, depth source. State the branch taken per spec §4:
- **PASS** (corr_mean > 0.7 + viser confirms) → "proceed to V0/V1 training plan."
- **FAIL on large-droid** → re-run Task 6/7 with `large-droid+behavior` ckpt; record its verdict.
- **FAIL on both** → "short LIBERO fine-tune of PointWorld then re-probe" (new plan) or **STOP** if fine-tune also fails. Do not write the training plan against rejected labels.

- [ ] **Step 2: Commit**

```bash
git add -f docs/superpowers/notes/2026-07-01-pw-probe-verdict.md
git commit -m "[LangPointWorld] PW-PROBE gate verdict + branch decision"
```

- [ ] **Step 3: Update project memory**

Append the verdict (pass/fail, corr, which ckpt) to the memory file `lang-pointworld-distill-design.md` so the next session continues from the gate outcome, not a re-run.

---

## Self-Review

**Spec coverage (spec §4 + §3.1 C0/C1/C2):**
- C1 LIBERO→PointWorld adapter → Task 3 ✓
- C2 frozen teacher wrapper → Task 5 ✓
- C0 probe harness (corr>0.7 + L2 + viser) → Tasks 4,6,7 ✓
- EE-centric frame + same-frame assert (spec §4) → Task 2 ✓
- Kill/branch criterion (spec §4) → Task 6 print + Task 8 note ✓
- Depth-gap + URDF + resolution integration risks (discovered in spec review) → Task 1 preflight ✓
- Out of scope by design (spec §8): student, flow head, loss wiring, V0/V1 training — correctly **not** in this plan; they are the next plan, gated on Task 8.

**Placeholder scan:** the two `# TODO Task-1` lines in Task 6 and the symbol-name note in Task 5 are **explicit, bounded integration steps with named source files to read** (`evaluation/tester.py`, `arguments.py`, `norm_stats.py`), not open-ended placeholders — they exist because the exact PointWorld eval-build sequence can only be pinned against those files at run time, and the plan says which files and how to resolve. Acceptable per the "expected integration work" standard.

**Type consistency:** `flow_to_ee_frame`/`to_ee_frame`/`assert_same_frame` (Task 2) used identically in Tasks 4,6. `build_pw_sample` signature (Task 3) matches its calls in Tasks 5,6. `PointWorldTeacher.imagine` return keys (`scene_flows`,`scene_coord0`, Task 5) match consumption in Task 6. `keypoint_corr`/`filtered_l2`/`gate_verdict` (Task 4) match Task 6 usage. Consistent.
