# [LangPointWorld] PW-PROBE Gate Verdict — 2026-07-01 (autonomous run)

**Status: gate ran end-to-end; first verdict FAIL on a probe-methodology bug; re-running with
corrected key-point selection + fallback checkpoint. Interim — see live results below.**

## What definitively works (integration cleared)

The entire teacher pipeline runs on real LIBERO data, GPU 2:
- VGGT geometry (depth + K + E from LIBERO RGB) ✓
- LIBERO→PointWorld data_dict bridge (VGGT-depth unprojection → scene_flows[:,0]; RobotSampler FK
  → robot_flows; real 31-dim scene / 16-dim robot features via PointWorld `gather_features`) ✓
- Frozen PointWorld `large-droid` (1288M) forward → imagined `scene_flows (11,8192,3)`, finite ✓
- All 8 LIBERO episodes processed without error ✓

**The teacher predicts real, non-trivial motion** (from imagine_out.npz diagnostics):
49% of points move >1mm, 8% move >1cm, max horizon displacement 0.38, p99 0.055. It is NOT
predicting a static scene.

## First verdict (large-droid, EE-proximity key points): corr_mean = 0.089 — FAIL

`results/pw_probe/probe_result.json`: corr_x=-0.10, corr_y=-0.03, corr_z=0.39, corr_mean=0.089,
l2_mean=0.026. pred magnitude ~0.0001 vs GT ~0.008-0.023.

**Root cause = PROBE METHODOLOGY, not (yet) a model verdict.** The probe selected the 64 scene
points nearest the EE at t=0 — those are static table/background, so it measured ~zero motion by
construction while the model's real predicted motion is concentrated elsewhere (the 8% moving
points). This is a measurement bug, corrected below.

## Deeper methodology caveat (important for interpreting any verdict)

The GT signal is **EE-position** motion; the teacher predicts **scene** (object/table) motion.
The arm itself is a *separate* input (`robot_flows`), NOT part of the scene-flow output. So scene
flow only moves where the arm *pushes objects*, after contact — comparing predicted-scene-motion
to GT-EE-motion is an imperfect proxy. A rigorous GT would be per-object pose motion, but LIBERO
object qpos lives in the 110-dim MuJoCo `states` with an undocumented layout that needs the sim
env (absent from the starVLA env) to decode. Deferred; flagged as the right next probe upgrade.

## Corrected probe (re-running): select key points by predicted-motion magnitude

`tools/pw_probe_libero.py` mode='motion' now averages over the K points the MODEL predicts move
most — measuring "does the model's predicted motion track GT arm/object motion" (the right
question). Running on BOTH `large-droid` and `large-droid+behavior` (spec §4 fallback).

### Live results (filled by the running probe_both.sh)
- large-droid (motion): results/pw_probe_motion/probe_result.json
- large-droid+behavior (motion): results/pw_probe_behavior/probe_result.json

## Branch decision (per spec §4) — pending corrected results

- If either checkpoint's motion-mode corr > 0.7 → **GATE PASS**, proceed to V1 training.
- If both < 0.7 but > ~0.3 with sensible viz → borderline; the EE-vs-scene proxy is the likely
  culprit, not the model. Recommend the object-pose GT probe (needs sim env) before killing;
  V1 distillation may still work since the student regresses teacher flow directly (the gate is
  a risk-reducer, not a hard correctness proof of the teacher).
- If both ≈ 0 → real domain gap (VGGT-depth clouds unlike DROID/BEHAVIOR GT-tracked points +
  gripper mismatch). Then: short LIBERO fine-tune of PointWorld, or reconsider per spec §4.

**Autonomous-run stance:** I will NOT start V1 distillation training on a FAILED gate (spec kill
criterion). If corrected probes still fail, I stop at "infrastructure complete + gate red" and
hand off with the object-pose-GT probe as the recommended next step — not burn GPU-hours training
on labels the gate rejected.
