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

## Full-demo action sampling (2026-07-01, later) — corr DROPPED 0.48→0.31, but more informative

Changed the 11 action timesteps from "front 11 consecutive frames" (~7% of a 148-frame demo,
mostly static approach) to **equally-spaced across the WHOLE demo** (approach→grasp→move→place).
Scene motion richness rose sharply (ep0 moved>1cm: 9%→38.6%; pred mean disp ×2.3).

**Gate result: corr_mean 0.48 → 0.31; L2_mean 0.05 → 0.25 (×5).** Per-episode:

| ep | front-11 | full-demo | note |
|----|----------|-----------|------|
| ep0 | 0.99 | 0.39 | strong→mid |
| ep1 | 0.38 | 0.56 | up |
| ep2 | -0.16 | 0.37 | up (was anti) |
| ep3 | 0.11 | 0.22 | up |
| ep4 | 0.37 | 0.56 | up |
| ep5 | -0.11 | 0.53 | up (was anti) |
| ep6 | 0.85 | 0.01 | strong→collapse |
| ep7 | 0.81 | 0.41 | strong→mid |

**Interpretation (the real signal):** full-demo **reduced variance** (no more 0.99/-0.16 extremes)
and lifted every previously-weak episode, but pulled the strong ones down → lower mean. The key
diagnostic is magnitude: GT EE displacement grew ×10 (long curved full-task trajectory) but the
teacher's predicted motion only grew ×3 → **the teacher UNDER-DRIVES large/long-range motion** on
LIBERO (same class of under-drive seen in [[vla-rlds-closedloop-rootcause]]). The front-11 0.48 was
partly inflated: short, near-linear EE motion is trivially easier to correlate than the full
approach→grasp→move→place arc. So neither number clears 0.7; full-demo is the more honest, harder
test and reveals under-drive as the core failure mode, not a fixable sampling artifact.

**Updated recommendation:** the EE-motion-vs-scene-motion proxy AND under-drive both cap this
gate. Two paths remain the right calls: (1) object-pose GT probe (needs sim env) to remove the
proxy mismatch; (2) pilot V1 and let closed-loop success decide — a distillation loss can still
transfer *direction* even if the teacher under-drives *magnitude* (normalize/whiten flow before
the distill loss). Under-drive is exactly the kind of bias a magnitude-normalized distill target
sidesteps. Still NOT a clean gate pass → not auto-training.

## 2026-07-02 — 诊断闭环 + 决定性对照(domain gap 实锤)

用户连问 EEF/proprio + 可视化 flow 飘,引出完整根因链,最终用官方数据对照实锤:

**根因链(全部验证):**
1. PW 不吃 EEF/proprio,机器人=无标签点云(用户直觉对)
2. flow 飘 = VGGT场景点云 与 FK机械臂点 坐标系脱节(场景z→2.0, 臂悬在0.2, 距离0.44)
3. 修复:LIBERO sim 渲染真米制深度+真K/E+object poses(libero env)→ build_from_sim → 机械臂站进场景(距离0.44→0.031), flow从臂发出
4. 残留:被抓物体 PW预测0.009 但 GT位移0.484 → 怀疑漏抓取
5. **决定性对照:官方 BEHAVIOR 数据上 PW l2=0.00106(毫米级),LIBERO ~0.25(差200倍)**

**结论:PW 模型优秀(in-domain 毫米级),pipeline已修对,唯一问题=LIBERO domain gap。**

**官方数据结构(下载验证):** scene_flow = per-object mesh点 × GT位姿轨迹(致密干净,被操作物体一定有flow)。我们LIBERO=深度反投影+PW预测(不准)。

**V1 正解(现在well-grounded):微调 PW on LIBERO。** 条件已具备:sim-render 出 per-object 6D poses + 米制深度 → 可做成官方格式(per-object mesh点 × GT位姿轨迹)微调数据。微调后 teacher 在 LIBERO 准 → 蒸馏。工具: tools/pw_sim_render.py, tools/pw_official_viz.py; 官方数据 /workspace/tingting/pw_official_data/; data分支 /workspace/tingting/PointWorld-data。
