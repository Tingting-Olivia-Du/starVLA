# [LangPointWorld] Phase-0 Preflight — go/no-go for LIBERO → PointWorld

**Date:** 2026-07-01 (overnight autonomous run)
**Verdict:** ✅ **GO** — PointWorld's full stack runs in the starVLA env; LIBERO can be adapted to its input contract. All three hard unknowns from spec review are resolved.

---

## 1. Environment (resolved — no rebuild needed)

The shared starVLA env (`/workspace/ghsun/miniconda3/envs/starVLA`, torch 2.6.0+cu124) was
**NOT mutated**. All PointWorld-only deps installed into an **isolated site-dir** under
`/workspace/tingting/` and injected via `PYTHONPATH`:

```
PW_EXTRA_SITE=/workspace/tingting/envs/pw_extra_site
export PYTHONPATH="$PW_EXTRA_SITE"      # prepend before running anything PointWorld
```

Installed there (all `--no-deps`, additive, reversible): `addict, pytorch_kinematics 0.10.0,
urdfpy 0.0.22, arm_pytorch_utilities, pytorch_seed, lxml, spconv-cu120 2.3.6, pccm, cumm-cu120,
pybind11, fire, ccimport, portalocker, lark, torch_scatter 2.1.2+pt26cu124, webdataset,
braceexpand, numba 0.65.1, llvmlite`.

**Verified imports** (torch 2.6, GPU 2): `pointworld.base.BaseModel`, `robot_sampler.RobotSampler`,
`dataset_components.pipeline.apply_release_pipeline_to_sample` — **full stack ready**.
`torch_scatter` must be the `pt26cu124` wheel; `spconv-cu120` works against the cu124 runtime.

**Why not the shared env or a fresh env:** additive isolated site-dir is the least-disruptive
choice on a shared multi-tenant box (spec/session safety constraint) and avoided a
`torch-2.12` pull that plain `pip install pytorch_kinematics` triggered.

## 2. Checkpoints (downloaded)

`HF_HOME=/root/angli/hf_cache`. All three present:
- `.../large-droid/model-best.pt` (13G) — **primary probe target**
- `.../large-droid+behavior/model-best.pt` (13G) — **fallback if gate fails**
- `.../small-droid/model-best.pt` (1.8G)

Snapshot dir: `/root/angli/hf_cache/hub/models--nvidia--PointWorld_models/snapshots/b9e2e19a4f2bd65922e1f6d70aa953fe70aa9dba/`.

## 3. Depth source (DECISION: VGGT primary, Depth-Anything fallback, sim-depth deferred)

LIBERO HDF5 obs has **no depth** (keys: `agentview_rgb, eye_in_hand_rgb, ee_pos, ee_ori,
ee_states, gripper_states, joint_states`; `actions`; `states (T,110)`), 128×128, `opengl`
image convention (⇒ agentview stored upside-down — adapter flips).

Options weighed:
- **Sim depth** (most accurate): LIBERO env *supports* `camera_depths=True` and the HDF5 stores
  full `states (T,110)` + `init_state` + `model_file` ⇒ demos are exactly replayable. **But
  mujoco/robosuite/libero are NOT in the starVLA env** ([[starvla-libero-eval-setup]]: LIBERO
  runs in a separate `libero-ttd` env via client-server). Cross-env cache generation = friction.
  **Deferred** to a borderline-gate fallback.
- **VGGT** (CHOSEN primary): `/workspace/tingting/vggt-world` is editable-installed in the
  starVLA env ([[geo-memoryvla-conda-env]]); imports OK. Produces metric point maps **AND joint
  camera intrinsics/extrinsics** from RGB → solves depth *and* the missing-intrinsics problem in
  one model, in-env, on GPU 2,3. Consistent with the Geo-MemoryVLA geometry backbone.
- **Depth-Anything** (fallback): `transformers 4.57` has `DepthAnythingForDepthEstimation`.
  Depth-only (still needs intrinsics from elsewhere). Simpler but less complete than VGGT.

**Engineering rationale (per user "choose most effective, not laziest"):** VGGT gives the most
complete geometry (depth + camera params) needed by PointWorld's `scene_featurizer`, avoids
cross-env friction, runs on GPU 2,3. The PW-PROBE gate is exactly the instrument to catch any
resulting depth error — if VGGT depth degrades the gate, switch to sim depth (cross-env) before
declaring the whole path dead.

## 4. Domain + URDF (DECISION: domain="droid", absolute-path the Franka URDF)

`resolve_robot_urdf("droid")` → `assets/franka_description/franka_panda_robotiq_2f85.urdf`
(relative to PointWorld root; **exists** at `/workspace/tingting/PointWorld/assets/...`).
- Arm: **7-DoF Franka Panda** — matches LIBERO's Panda arm exactly ⇒ arm point-flow FK is valid.
- Gripper: PointWorld URDF is **Robotiq 2F85**, LIBERO is Franka's parallel gripper — differs.
  Documented approximation (spec §4 domain gap); arm dims are the gate's key points anyway,
  matching the FCDecoder-probe methodology ([[vla-rlds-closedloop-rootcause]]).
- Consequence for code: run teacher from PointWorld root OR pass the URDF as an absolute path so
  the relative `assets/...` resolves.

## 5. Resolution

PointWorld pipeline asserts camera payload `(180, 320)`; LIBERO is 128×128. Adapter resizes
(INTER_AREA for RGB, INTER_NEAREST for depth) — implemented in `libero_camera_adapter.py`.

## 6. Symbol corrections discovered (feed into teacher wrapper, plan Task 5)

- `pointworld.norm_stats` exposes `load_norm_stats_from_json` / `load_stats_from_json_folder`
  (NOT `load_norm_stats` as the plan guessed). Use `stats/droid` folder (present in repo).
- Model build path is `BaseModel(args, data_info_dict, rank)`; real build+load sequence is in
  `evaluation/tester.py:__init__` (torch.load `weights_only=False`, key `model`) — mirror it.

## 7. CRITICAL ARCHITECTURAL FINDING — PointWorld consumes preprocessed points, not RGB-D

Discovered while wiring `imagine()`. **PointWorld's `main` branch does NOT unproject RGB-D.**
`apply_release_pipeline_to_sample` and `sample_cameras` both *assume the sample already contains*
`<cam>_scene_flows` — full-scene 3D point **trajectories** `(T,N,3)` computed OFFLINE by the
`data` branch via **mesh + per-timestep object-pose tracking** (`decoders.py:transform_points_kernel`
transforms local mesh points by GT object poses). This is GT scene tracking, NOT monocular depth
unprojection. The `data` branch (which does this) is a separate branch, not checked out.

**Consequence for the probe (the bridge in `libero_to_datadict.py`):** BaseModel.forward only
*conditions* on `scene_flows[:,0]` (t=0 cloud) + `robot_flows` (all T); the future scene frames
are what it PREDICTS. So the minimal runnable data_dict is:
- `scene_flows[:,0]` = t=0 cloud via **VGGT-depth unprojection** (K^-1 backproject + E^-1 to world);
  future frames = dummy zeros (predicted output). Feature encoding (DINOv3) is computed inside
  forward from this cloud + the camera RGB.
- `robot_flows[T,Nr,3]` = FK over the demo joint trajectory via PointWorld `RobotSampler`
  (`_get_robot_flows_droid`, needs `joint_positions[T,7]` + `gripper_positions[T]`, droid Franka
  URDF). LIBERO: `joint_states[T,7]` + `gripper_states[T,2]` (mean → single finger proxy).

**Implication for V1 distillation (not just probe):** the teacher's "scene_flows" targets on
LIBERO will be depth-unprojected clouds propagated by the model, NOT GT mesh-tracked trajectories
like DROID/BEHAVIOR training data. This is an ADDITIONAL domain-gap axis on top of robot/camera.
The PW-PROBE gate measures exactly this — if the teacher's imagined flow on VGGT-depth LIBERO
clouds doesn't track GT (EE-trajectory proxy), the gate fails and we branch per spec §4. This is
the single biggest open risk and is precisely what the gate exists to answer cheaply.

**Status at handoff:** teacher loads (1288M), VGGT geometry works, all 6 langpw unit modules
pass (18 tests). The `libero_to_datadict` bridge is written but the exact camera_data key naming
the DINOv3 featurizer expects (`_extract_camera_data`) still needs a GPU iteration to pin — this
is the next concrete step to a real imagine() forward, then the gate verdict.

## 8. Model sequence length T=11 (CONTEXT_HORIZON=1 + PRED_HORIZON=10)

pointworld/base.py:32-33: `CONTEXT_HORIZON=1, PRED_HORIZON=10` → `self.T=11`. The model takes
t=0 context and predicts T-1=10 future frames. **The bridge horizon MUST be 11** — scene_features
dim is `scene_flows(3)+scene_colors(3)+scene_normals(3)+gripper_open(T)+dist2robot(T)` = 9+2T = 31
at T=11 (was 25 at T=8, a shape-mismatch crash). robot_features=16 is T-independent
(flows3+colors3+normals3+gripper1+vel3+accel3). This T=11 is fixed by the checkpoint; the probe
and cache both use horizon=11 and read out the 10 predicted future frames.
