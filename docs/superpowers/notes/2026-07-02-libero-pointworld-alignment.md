# LIBERO → PointWorld Alignment — Full Record

**Date:** 2026-07-01 → 2026-07-02
**Context:** Feeding LIBERO episodes to the frozen PointWorld world model (teacher) and visualizing
its imagined scene-flow with the official `PredictionVisualizer`. This note records the complete
diagnosis + every alignment fix, so the pipeline is reproducible and the residual (pure domain
gap) is cleanly isolated for the V1 finetune.

Marker for all code: `[LangPointWorld]`. Env: run PointWorld/VGGT things with the starVLA python +
`PYTHONPATH=/workspace/tingting/envs/pw_extra_site:/workspace/tingting/starVLA`; run sim-render
with the `libero` conda env (see §4).

---

## 0. TL;DR

The "flow floats in mid-air / arm mesh separated from point cloud" artifacts were **coordinate-frame
bugs, not model bugs**. PointWorld is near-perfect on its own BEHAVIOR data (l2 ≈ 0.00106,
millimeter) and only fails on LIBERO because of (a) VGGT-vs-FK frame misalignment and (b) a
base-frame mismatch between our robot points and the official URDF mesh. After the sim-render +
base-frame fixes, the robot stands in the scene (robot-to-scene 0.44 → 0.028) and any residual is
the genuine LIBERO domain gap → the target of the V1 PW-finetune.

---

## 1. What PointWorld consumes (verified)

`BaseModel.forward` reads ONLY: `scene_flows` (scene 3D points, t=0 conditioned; future predicted),
`scene_features` (31-dim), `robot_flows` (robot 3D point cloud, all T frames = the action),
`robot_features` (16-dim), `robot_exists`, `__domain__`. **No EEF pose, no proprio/joint channel.**
The robot is an *unlabeled* point cloud; EEF is implicit. Scene input is **multi-camera** capable
(`scene_featurizer`: `C=len(cams)`, cam0/cam1/...). Model T = 11 (CONTEXT_HORIZON=1 + PRED_HORIZON=10).
Scene_features dim = `scene_flows(3)+scene_colors(3)+scene_normals(3)+gripper_open(T)+dist2robot(T)`
= 9 + 2T = 31 at T=11.

Official scene_flows = per-object **mesh points × GT pose trajectory** (NOT depth unprojection);
so the manipulated object always has GT flow. Our LIBERO scene = depth unprojection + PW prediction.

---

## 2. The bug chain (all verified with data)

1. **PW takes no EEF/proprio** → robot is an unlabeled cloud (user's intuition, confirmed).
2. **VGGT-vs-FK frame misalignment (flow floats).** Original bridge used VGGT-estimated depth (scale
   arbitrary, z→2.0) + VGGT extrinsic = **identity** (camera = origin). FK robot points were in the
   URDF base frame (z 0.20–0.36). Result: scene cloud and robot cloud in different frames/scales;
   **robot-to-nearest-scene-point distance = 0.44**, arm floating below the scene, touching nothing
   → PW predicts motion in the wrong place / under-drives.
3. **Sim-render fix** (§4): LIBERO sim replay gives real metric depth + real K/E + object poses, all
   in ONE world frame → robot-to-scene **0.031**, arm stands in the scene.
4. **Grasped-object miss.** Even aligned, PW predicted ~0.009 motion near the grasped object while
   its GT displacement was 0.484 → suspected PW misses grasp causality.
5. **Decisive control.** PW on official BEHAVIOR WDS: **l2/mean = 0.00106 (mm), conf 0.9999**, vs
   LIBERO ~0.25 (×200 worse). ⇒ PW model is fine in-domain; LIBERO failure is **pure domain gap**.
6. **Robot mesh / flow separation (viz).** Official viz renders the arm from the URDF FK'd at the
   **base origin (0,0,0)**, but our `robot_flows` were transformed to the true LIBERO base at world
   **(-0.6, 0, 0)** → mesh and flow separated by ~0.6 in x. **Base-frame fix** (§5).

---

## 3. Real LIBERO camera / base parameters (no sim env needed to read these)

From `LIBERO/libero/libero/envs/bddl_base_domain.py:_setup_camera` and robosuite:
- **agentview (fixed cam):** pos `[0.5886131746834771, 0.0, 1.4903500240372423]`,
  quat(wxyz) `[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]`,
  fovy = 45° → K: `f = 0.5*H/tan(fovy*π/360) = 154.5` for 128px, cx=cy=64. (Matches sim-rendered K.)
  mujoco cam looks −z: `R_c2w = quat2R(q) @ diag(1,-1,-1)` for OpenCV; E(world→cam) = `[R_w2c | -R_w2c@pos]`.
- **wrist / eye_in_hand (on eef):** robosuite panda `robot.xml`: pos `[0.05,0,0]`, quat `[0,0.707,0.707,0]`,
  **fovy=75**; world pose = FK(eef link) @ this offset, PER-FRAME.
- **robot base in world = (-0.6, 0, 0)** (pure translation, no rotation), from sim `robot0_base` pose.
  (robosuite `base_xpos_offset["table"] = (-0.16 - table_len/2, 0, 0)`, table_len=1 → -0.66; sim gives -0.6.)
- **LIBERO Panda URDF exists:** `.../libero/.../robosuite/models/assets/bullet_data/panda_description/urdf/panda_arm_hand.urdf`
  (mesh refs use `package://panda_description/...` → must be rewritten to absolute paths).

---

## 4. Sim-render (the correct data source) — tools/pw_sim_render.py

Runs in the **`libero` conda env** (py3.8, robosuite 1.4.1, mujoco 3.2.3), NOT starVLA. h5py in an
ISOLATED dir `/workspace/tingting/envs/libero_extra_site` (inject via PYTHONPATH; robosuite
`camera_utils` imports h5py so it must be on path).

```
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=<gpu> \
PYTHONPATH=/workspace/tingting/envs/libero_extra_site \
/workspace/ghsun/miniconda3/envs/libero/bin/python tools/pw_sim_render.py \
  --hdf5 <libero_episode.hdf5> --demo demo_0 --out <sim_render.npz>
```

Replays HDF5 `states` in robosuite → renders **real metric depth** + real K/E + **per-object 6D poses**.
Gotchas (all handled in the tool):
- `OffScreenRenderEnv(**kw)` — pass ONLY `bddl_file_name` + `camera_*` (full env_kwargs dupes
  `controller_configs` → TypeError).
- bddl path in HDF5 is stale → resolve to `/workspace/tingting/LIBERO/libero/libero/bddl_files/<suite>/<name>`.
- robosuite depth is **normalized [0,1]** → metric via near/far:
  `near=znear*extent, far=zfar*extent, metric=near/(1-d*(1-near/far))`.
- depth image is **vertically flipped (opengl)** → `d[::-1,:]` before unproject.

**Output npz keys:** `ti` (linspace indices), `obj_names` (25: floor/table/robot0_base/robot0_link0-7
+ gripper links + task objects), `agentview_{rgb,depth,K,E_cam2world}`, `robot0_eye_in_hand_{...}`,
`obj_poses [T,25,4,4]`. `E_cam2world` is 4×4 cam→world. Robot link world poses are given directly
(no FK needed for robot points); per-object poses = the rigorous gate GT.

**Alignment verified (ep0, after vertical flip):** scene cloud z 0.00–0.49, robot links z 0.00–0.66,
objects z 0.00–0.36, table = point-cloud z-mode 0.01 — all one metric world frame.

---

## 5. Bridge: build_from_sim — starVLA/model/modules/langpw/libero_to_datadict.py

`LiberoDataDictBuilder.build_from_sim(sim_npz, joint_states, gripper_states, horizon,
cams=("agentview","robot0_eye_in_hand"))` builds the PW `data_dict`, **everything in the ROBOT-BASE
frame** (base at origin) so it matches the official viz URDF mesh (FK'd from base origin):

- **Multi-view scene cloud:** for each cam, unproject its t=0 real depth with real K/E (cam→world),
  then **world → base** (`pts = base_inv[:3,:3] @ pts + base_inv[:3,3]`), workspace-crop, and MERGE
  both cams' clouds (they share the sim world frame → aligned). Emits `cam0_*`/`cam1_*` for PW's
  multi-view encoder. Camera payload extrinsic = **base→cam = E_w2c @ base_T** (points are base-frame).
- **Robot cloud:** FK points are ALREADY in the base frame — **kept as-is** (no base_T transform),
  so they align with the official URDF mesh. (Earlier bug: we applied base_T → moved robot to world
  (-0.6) while the mesh stayed at origin → separation.)
- **Depth flip** (`d[::-1,:]`), scene/robot features via PointWorld `gather_features` (real 31/16-dim).

**Results (dual-view, base-frame):** merged cloud ~7–8k pts, robot-to-nearest-scene **0.028**,
scene z 0.00–0.49, robot z 0.20–0.36. Dual-view cam positions (base frame): cam0 (agentview)
`[1.497, 0, 0.65]`, cam1 (wrist) `[0.517, -0.027, 0.356]` (≈ eef, orientation looks straight down,
axis·(to-eef) = 0.89 — cam1 verified correct; the "cam1 not near eef" impression was the wrist
frustum being small + occluded by the arm mesh).

---

## 6. Official-viz driving (headless) — tools/pw_official_viz.py, tools/pw_viz_official_data.py

- `pw_official_viz.py` drives the official `PredictionVisualizer` on OUR cached sample (LIBERO
  sim-aligned) and HOLDS the viser server (the eval loop otherwise blocks on `input()` and closes
  the session). Flags: `--cache <npz>` (arbitrary sample, e.g. dual-view), `--urdf <path>`,
  `--viewer-port`. Reads `cam0/cam1_*`, `scene_flows`, `scene_colors`, `robot_flows`,
  `joint_positions`, `scene_supervised_mask` (all present in the sim sample).
- `pw_viz_official_data.py` = same but for OFFICIAL BEHAVIOR WDS data (replicates
  `tester._visualize_eval_samples` without the TTY prompt; `VIZ_SCAN` picks the max-GT-motion
  window). Official data → official rendering (matches the paper's project-page viz).
- **Official BEHAVIOR data prep:** download `nvidia/PointWorld-BEHAVIOR` task-0000 (1.66GB) →
  `cat part-* | zstd -d | tar x` (python `zstandard` streaming) → 2.9G of `episode_*.hdf5` at
  `/workspace/tingting/pw_official_data/behavior_restored/`. data-branch pipeline (cloned to
  `/workspace/tingting/PointWorld-data`): `data_integrity_check.py` → `make_wds_manifest.py`
  (`--max_clips`, `--test_percentage` in [0,1]) → `convert_wds.py` → WDS at
  `.../behavior_wds_big/` (200 clips, test=60).

### URDF caveat (LIBERO Panda vs Robotiq)
Official viz mesh render **hardcodes the Robotiq joint name `finger_joint`**
(`viser_flow/robot_sampler_lite.py:108`, `robot_flow.py:75`, `visualizer.py:954`). Swapping in the
LIBERO `panda_arm_hand.urdf` (mesh paths rewritten to absolute, saved at
`starVLA/assets/libero_panda/panda_arm_hand.urdf`) → `KeyError: 'finger_joint'` because LIBERO uses
`panda_finger_joint1/2`. **The 7 arm links are identical Franka Panda in both URDFs; only the
gripper differs.** So current viz uses the PointWorld Robotiq URDF (arm aligns correctly after the
base-frame fix; only the gripper shape is Robotiq-not-Panda). Full LIBERO-URDF swap needs joint-name
patches to those 3 official-viz spots — deferred (cosmetic, doesn't affect alignment).

---

## 7. Current running viewers (session)

- `localhost:8080` — official BEHAVIOR data + official rendering (baseline / standard answer).
- `localhost:8084` — LIBERO dual-view (cam0 agentview + cam1 wrist), base-frame aligned, official
  rendering, PointWorld Robotiq URDF. Cache: `/workspace/tingting/.tmp/sim_dualview_base_ep0.npz`.

viser sometimes hops ports if the requested one is busy (banner shows the real port). Forward the
port shown in the banner over SSH (`ssh -L <port>:localhost:<port> ...`).

---

## 8. Residual = the V1 target

After all alignment fixes, PW still (a) under-drives the arm's own motion and (b) misses the
grasped-object motion on LIBERO — the genuine domain gap (PW is mm-accurate on BEHAVIOR). **V1 =
finetune PW on LIBERO.** We already have what's needed: sim-render gives per-object 6D poses +
metric depth, so LIBERO can be built into the official per-object-mesh × GT-pose-trajectory format
for finetuning.

## Key files
- `tools/pw_sim_render.py` (libero env), `tools/_pw_sim_imagine_smoke.py` (starVLA env)
- `starVLA/model/modules/langpw/libero_to_datadict.py` → `build_from_sim` (base-frame, multi-view)
- `tools/pw_official_viz.py` (--cache/--urdf), `tools/pw_viz_official_data.py`, `tools/pw_official_to_render.py`
- `assets/libero_panda/panda_arm_hand.urdf`; official data at `/workspace/tingting/pw_official_data/`
