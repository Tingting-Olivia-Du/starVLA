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

### FINAL corrected results

**large-droid (motion-mode): corr_mean = 0.48 — BORDERLINE (below 0.7, but real signal).**
corr_x=0.69, corr_y=0.43, corr_z=0.32; l2_mean=0.050. Pred/GT magnitudes now well-matched
(~0.02 vs ~0.01 — no under-drive). Per-episode split is the key finding:

| ep | corr_mean | pred\|.\| | gt\|.\| | reads as |
|----|-----------|-----------|---------|----------|
| ep0 | **0.99** | 0.027 | 0.023 | strong track |
| ep6 | **0.85** | 0.025 | 0.015 | strong track |
| ep7 | **0.81** | 0.020 | 0.012 | strong track |
| ep1 | 0.38 | 0.014 | 0.008 | weak |
| ep4 | 0.37 | 0.026 | 0.011 | weak |
| ep3 | 0.11 | 0.018 | 0.008 | none |
| ep5 | -0.11 | 0.022 | 0.007 | anti |
| ep2 | -0.16 | 0.017 | 0.011 | anti |

**Interpretation:** the teacher captures real LIBERO manipulation dynamics on ~half the episodes
(strong, magnitude-matched tracking) and fails on the other half. This is NOT a dead teacher and
NOT under-drive — it is *partial* transfer, exactly the borderline the spec anticipated. The
noisy half is plausibly dominated by the **EE-motion-vs-scene-motion proxy mismatch** (§ caveat
above): scene flow ≠ arm motion, so episodes where the object barely moves until late contact
score poorly against an EE-motion GT even if the scene prediction is fine.

**large-droid+behavior (fallback): NOT EVALUATED — feature-layout incompatible.** It loads (after
the multi-domain norm-stats fix) but expects **42-dim** scene features (BEHAVIOR bimanual layout)
vs the 31-dim droid layout the bridge builds → per-episode dim-mismatch crash. Matching the 42-dim
BEHAVIOR feature composition is a bounded but non-trivial follow-up; deferred (large-droid is the
DROID-native, correct-feature checkpoint for a Franka-arm LIBERO probe anyway).

## FINAL branch decision (autonomous run)

**Gate = BORDERLINE, treated as NOT-a-clean-PASS → DID NOT start V0/V1 training.** This honors the
spec kill-criterion (don't train on a gate that didn't clearly pass) and the user's constraint to
not burn GPU-hours speculatively. The full V1 pipeline is built, tested, and ready to run the
moment the gate is cleared.

**Recommended next steps (ranked), for the user's return:**
1. **Object-pose GT probe (the rigorous fix).** Replace the EE-motion GT proxy with per-object
   3D motion from LIBERO sim object poses. Needs the `libero-ttd`/robosuite env (not in starVLA
   env) to decode object qpos from the 110-dim MuJoCo `states`, or to re-render with object
   segmentation. This directly tests "does teacher scene-flow track object motion" — the actual
   V1 supervision signal — and likely lifts corr well above the EE proxy on the strong episodes.
2. **Pilot V1 at a relaxed bar.** The 3 strong episodes (corr 0.8–0.99) show the teacher gives
   genuinely useful flow on many samples. A small pilot: build the cache, train V1 vs V0 on the
   libero_object suite, and let the CLOSED-LOOP success-rate delta be the real arbiter (the gate
   is a cheap proxy; the true test is whether distillation helps the policy). Weight the distill
   loss by per-sample teacher confidence so noisy-teacher samples contribute less.
3. **Fix the 42-dim BEHAVIOR feature layout** and probe large-droid+behavior — sim-trained
   BEHAVIOR may transfer to LIBERO-sim better than real-DROID.

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
