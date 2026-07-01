# Language-Grounded Point-Flow Distillation from a Privileged PointWorld Teacher — Design Spec

**Date:** 2026-07-01
**Framework host:** starVLA (`/workspace/tingting/starVLA`)
**Status:** Design approved, pending implementation plan
**Working title:** *Learning Language-Grounded Point-Flow Policies from a Privileged 3D World Model*
**Companion specs:**
- [`2026-06-30-d4rt-worldstate-design.md`](2026-06-30-d4rt-worldstate-design.md) — the controlled-experiment methodology this reuses (config switch + marker + byte-identical baseline + PROBE gate).
- [`2026-06-29-geo-memoryvla-design.md`](2026-06-29-geo-memoryvla-design.md) — the host policy (GR00T flow-matching head, dataloader, LIBERO eval) this attaches to.
- Source plan this refines: `/workspace/tingting/plan/pointworld-teacher-student.md`.

---

## 0. Coding convention (applies to ALL code in this project)

Every new file and modified block MUST be greppable. Use the marker `[LangPointWorld]`.

- **New file header:**
  ```python
  # [LangPointWorld] <one-line role of this file in the design>
  # Part of the PointWorld point-flow distillation arm — see
  # docs/superpowers/specs/2026-07-01-lang-pointworld-distill-design.md
  ```
- **Modified block in an existing file:**
  ```python
  # [LangPointWorld] <what was added/changed and why>
  ```
- When code is vendored from PointWorld (NVIDIA, Apache-2.0 for the released
  train/eval code — verify per-file; DINOv3 backbone is under a separate,
  access-gated license and is NOT redistributed here), note provenance in the header
  (`# Vendored from NVlabs/PointWorld (Apache-2.0): github.com/NVlabs/PointWorld`).

---

## 1. Research thesis & the controlled experiment

> **A pretrained RGB-D + action-conditioned 3D world model (PointWorld), used as a
> *privileged, frozen teacher* that predicts future 3D point-flow for demonstration
> actions, distills useful 3D-dynamics structure into an RGB-only, language-conditioned
> student policy — improving closed-loop manipulation over the identical policy trained
> with behavior cloning alone. The teacher is never run at deployment.**

Two roles, cleanly separated (the framing that makes this a paper rather than
"language-conditioned PointWorld"):

- **Teacher (PointWorld, frozen):** *"Given this action, how will the 3D world move?"*
- **Student (starVLA policy):** *"Given this language goal and RGB-only observation, how
  should the 3D world move, and what action causes that motion?"*

### 1.1 Controlled experiment (independent variable)

One config switch selects whether the distillation stream is active. Everything else
— Qwen-VL semantic stream, memory, GR00T flow-matching action head, LIBERO dataloader,
eval harness — is **byte-identical**.

`framework.pw_distill.enabled ∈ {off, on}`

| Held constant (shared) | Varied (the independent variable) |
| --- | --- |
| starVLA policy (`GeoMemoryVLA` host), GR00T flow-matching action head, dataloader, LIBERO eval harness, action config | **off:** behavior-cloning baseline (V0) <br> **on:** + student point-flow head supervised by cached PointWorld teacher flow (V1) |

Any closed-loop LIBERO success-rate delta between `off` and `on` is attributable to the
PointWorld distillation signal, because nothing else differs.

### 1.2 What runs when (the most-misunderstood point — state it explicitly)

**Training time — both present:**
- The **student policy** `π_θ(image, language, proprio) → action` learns by ordinary
  behavior cloning (flow-matching action loss), exactly like the current VLA.
- A **frozen PointWorld** was run **offline, once**, over demonstration actions to
  produce cached future 3D point-flow targets. The student grows a **point-flow head**
  that regresses those cached targets. PointWorld itself does **not** run in the training
  loop in V1 — only the cache is read.

**Test time — student only:**
- `action = π_θ(current image, language, proprio)` — one forward pass. PointWorld, the
  teacher cache, and the point-flow head's dependence on the teacher are **not loaded**.
  Closed-loop LIBERO speed is unchanged from the V0 baseline. The world-model knowledge
  has been distilled into `θ`.

> Analogy: PointWorld is a **coach** who corrects your motion in training and is absent at
> game time; the **player (policy)** produces every action, in training and at test.

---

## 2. Verified facts about PointWorld (drive every design decision)

Verified against `/workspace/tingting/PointWorld` code + README, not assumed.

| Fact | Value | Source | Design consequence |
| --- | --- | --- | --- |
| Model class | **action-conditioned 3D dynamics model, NOT a policy** | README, `pointworld/base.py` | Cannot be deployed as a policy; used as a teacher only. |
| Conditioning | `(scene_coord0, scene_feat0, robot_coord_seq, robot_feat, robot_exists)` → future scene point-flow. **No language pathway anywhere.** | `pointworld/base.py:242,434` | "Language-conditioned" must be added by the *student*, not PointWorld. |
| Scene input | RGB **+ depth + intrinsics + extrinsics** per camera | `scene_featurizer.py:160-163` | LIBERO adapter must supply RGB-D + camera params (sim provides all). |
| Action representation | robot action → **3D robot point-flow via differentiable FK** (urdfpy PK chain) | `robot_sampler.py:21-24`, `utils.build_pk_chain_from_urdf` | V1 only needs FK *forward* (offline). Differentiable FK is a V3 concern (see §7). |
| Output | full-scene future 3D point-flow `scene_flows [B,T,Ns,3]` (relative displacement + base coord) | `pointworld/base.py:483-487` | Student distills **sparse** task-relevant subset, not dense flow (§5.3). |
| Pretraining domains | **DROID (real Franka + Robotiq)** and **BEHAVIOR (sim)** | README | **LIBERO is out-of-domain** (different robot/gripper/cameras) → domain-gap risk gates V1 (§4). |
| Released checkpoints | `small-droid`, `large-droid`, `large-droid+behavior`, `filter_droid_test_split` | HF `nvidia/PointWorld_models` | Probe `large-droid` first; `large-droid+behavior` is the fallback if the gate fails (§4). |
| Paper metric | `full_eval/test/filtered_l2_moved/mean` (expert-confidence-filtered L2 on moved points) | README §Eval | Reuse as one PROBE gate metric; needs the released confidence artifact for DROID only, not LIBERO. |
| License | released train/eval code Apache-2.0; **DINOv3 backbone access-gated, separate license** | README §Third-Party | DINOv3 weights NOT redistributed; document as an external download dependency. |

---

## 3. Architecture — attach a distillation stream to the existing host

The host `GeoMemoryVLA.forward()` already returns `{"action_loss": ...}` and optionally
adds `{"imagination_loss": scale * imag_loss}`
([GeoMemoryVLA.py:345-348](../../../starVLA/model/framework/VLM4A/GeoMemoryVLA.py)).
The distillation stream slots in at **exactly that seam**: a new
`{"pw_distill_loss": λ * flow_loss}` term, computed by a new head, gated by
`framework.pw_distill.enabled`. The action head, cond assembler, memory, dataloader, and
eval change by **~30 LOC of wiring + one new head module + one offline cache script**.

```
TRAINING
  image_ext[t-K:t], image_wrist[t-K:t], language, proprio
        │                                   (existing student streams: Qwen-VL + memory)
        ▼
  cond  ──────────────────────────────────────────────► GR00T flow-matching head ─► action_loss   (UNCHANGED)
        │
        ├──► [LangPointWorld] Point-Flow Head ─► F_student [B,N,Hf,3]
        │                                             │
        │                       cached teacher target F_teacher [B,N,Hf,3]  (offline PointWorld)
        │                                             ▼
        │                                   L_flow = Σ w_i ρ(F_student − F_teacher)
        ▼
  out = {action_loss,  pw_distill_loss = λ · L_flow}

TEST (LIBERO closed loop)
  image, language, proprio ─► student ─► action     ← PointWorld / cache / flow-head-target NOT loaded
```

### 3.1 New components (each independently testable)

| # | Component | File (new unless noted) | Role | Depends on |
|---|---|---|---|---|
| **C0** | **PW-PROBE gate** | `tools/pw_probe_libero.py`, `tools/pw_probe_visualize.py` (mirror `d4rt_probe_libero.py` / `d4rt_probe_visualize.py`) | Frozen PointWorld on LIBERO demo actions → compare imagined vs GT future flow. **Runs before any training.** §4. | PointWorld ckpt, LIBERO HDF5, viser |
| **C1** | **LIBERO→PointWorld camera adapter** | `model/modules/langpw/libero_camera_adapter.py` | LIBERO RGB-D + sim intrinsics/extrinsics → PointWorld `camera_data` contract. | `scene_featurizer.py` input schema |
| **C2** | **PointWorld teacher wrapper** | `model/modules/langpw/pointworld_teacher.py` | Load frozen `large-droid` (or `+behavior`) ckpt; `(rgbd, demo_action_chunk) → future 3D flow`. FK forward via vendored `robot_sampler`. | PointWorld `base.py`, `robot_sampler.py`, HF ckpt |
| **C3** | **Offline teacher-flow cache** | `tools/pw_teacher_cache.py`, `model/modules/langpw/point_sampling.py` | Per-sample: run C2, sample sparse points (§5.3), save `teacher_flow / teacher_points / visibility / confidence`. **Where PointWorld runs — once, offline.** | C1, C2 |
| **C4** | **Student point-flow head** | `model/modules/langpw/point_flow_head.py` | `cond → F_student [B,N,Hf,3]` (+ optional visibility/contact). Feeds tokens back into action head cond (§5.4). | host `cond` |
| **C5** | **Loss + config wiring** | edit `GeoMemoryVLA.py` (~30 LOC, `[LangPointWorld]`) + `pw_distill_config` dataclass | Gate on `framework.pw_distill.enabled`; add `λ · L_flow` to the returned dict, mirroring `imagination_loss`. | C4, cache |

### 3.2 Interfaces (the boundaries that keep units isolated)

- **C3 → training**: a frozen, on-disk cache keyed by `(episode_id, timestep)`. The
  training loop never imports PointWorld — it reads `.npy`. This is the isolation that
  removes PointWorld's cost and differentiability from the V1 critical path.
- **C4 → action head**: point-flow tokens are appended to `cond` before the GR00T head
  (§15.4 of the source plan — flow must *influence* action, not just be an aux loss).
  Guarded by `pw_distill.feed_flow_tokens` so the "aux-loss-only" ablation is one switch.
- **C0**: standalone, imports C1+C2 only. No training dependency. Its verdict is a
  **hard gate** on whether V1 proceeds (§4).

---

## 4. PW-PROBE gate — the cheapest falsification (RUNS FIRST)

The one hard risk in V1 is **distilling wrong labels**: if PointWorld's imagined flow on
out-of-domain LIBERO is wrong, the student learns garbage. The source plan (§Risk 1)
names this risk but has no gate. We add one, mirroring the D4RT `d4rt_probe_libero`
success pattern (numbers + teacher-forced visualization cross-check).

**Procedure (zero training):** freeze PointWorld; for a set of LIBERO demo episodes, feed
the **real** demo actions (→ FK → robot-flow) and the t=0 RGB-D scene; compare the
**imagined** future scene-flow against the **GT** future flow (from sim depth across
frames), on task-relevant points (gripper + moved-object points).

**Gate metrics (both required) + visualization:**
1. **Per-dim correlation** of imagined vs GT displacement on key points — sensitive to
   direction/sign errors, tolerant of scale offset. **Gate: key-point corr > 0.7.**
   (Same instrument that surfaced FCDecoder under-drive.)
2. **Filtered L2 endpoint error** (PointWorld's own `filtered_l2_moved` style, in a
   consistent frame) — absolute-scale sanity. **Gate: report + sanity threshold TBD from
   the first probe run's distribution; not a blind cutoff.**
3. **Viser visualization** of imagined vs GT flow on ≥8 episodes — human confirmation the
   numbers aren't lying (mirrors `d4rt_probe_visualize.py`).

**Coordinate-frame discipline (must be pinned before the probe means anything):** teacher
flow, GT flow, and later student flow must live in **one** frame. V1 default:
**robot-base frame** (LIBERO sim gives reliable camera extrinsics, so
`T_camera_to_base` is exact — the source plan's §9.3 concern is a non-issue in sim).
Object-relative is a V2+ robustness upgrade.

**Kill / branch criterion (explicit):**
- Gate **passes** → proceed to V0/V1.
- Gate **fails** on `large-droid` → retry with `large-droid+behavior` ckpt.
- Still fails → **short LIBERO fine-tune of PointWorld** (reuse existing RLDS/HDF5
  pipelines) then re-probe.
- Fails after fine-tune → **this distillation path is not viable on LIBERO; stop.**
  Do not train on labels the gate rejected. Document and reconsider (e.g. a
  RoboMimic/DROID-adjacent benchmark, or the online-reward variant §7 with a LIBERO-tuned
  world model).

---

## 5. Student details (V1 scope)

### 5.1 Inputs (reuse host; no new dataloader)
Dual-view RGB temporal window `K∈{2,4}`, language, proprio window — all already available
to `GeoMemoryVLA`. **No RGB-D at student input** (RGB-D is teacher-only, offline).

### 5.2 Recommended starting values
`K=2`, action horizon `Ha` = host default, flow horizon `Hf∈{4..8}`, sparse points
`N∈{128,256}`.

### 5.3 Point selection (hybrid, `N=256`)
`96` target-object · `48` target-region · `48` gripper/contact · `64` background/context.
Object/target/gripper masks from LIBERO sim segmentation (available for free in sim) →
sidesteps the source plan's "needs pseudo-labels" caveat for V1. Learned/patch-anchored
points (plan's B1/B2) deferred to real-robot benchmarks where masks are unavailable.

### 5.4 Flow→action coupling
Point-flow tokens appended to `cond` feeding the GR00T head, guarded by
`pw_distill.feed_flow_tokens`. **Scheduled teacher-forcing is deferred** (source plan
§15.5) — a known train/test-drift tuning hazard; V1 uses student-predicted flow tokens
throughout and treats the coupling on/off as a clean ablation.

### 5.5 Loss (V1)
`L = L_action + λ_flow · L_flow`, with
`L_flow = Σ_{i,h} w_i · ρ(F_student[i,h] − F_teacher[i,h])`, `ρ` = smooth-L1,
`w_i` up-weighting moving/object/gripper points and down-weighting low-confidence/static
points (weights from the cache's confidence field). No other loss terms in V1.

---

## 6. Milestone ladder (risk-ordered)

| Stage | What | Milestone / gate | Kill criterion |
|---|---|---|---|
| **C0** | PW-PROBE gate | corr>0.7 on key points + L2 sane + viser confirms | see §4 branch/kill |
| **V0** | BC baseline (`pw_distill.enabled=off`) | reproduce a reasonable LIBERO success rate | baseline itself broken → fix host, not this project |
| **V1** | offline flow distillation (`on`) | **closed-loop success rate (or robustness) improves over V0** | V1 ≤ V0 after tuning λ_flow and feed_flow_tokens → distillation adds nothing; report negative, stop or pivot to V2 grounding |
| **V2** | language-grounded desired flow (object/target/motion queries + grounding loss) | model attends to correct object/target; task-flow meaningful | — |
| **V3** | *(= the original online differentiable-reward "Plan A")* inverse consistency: student action → **differentiable** FK → frozen PointWorld → consistency with desired flow | predicted actions cause teacher-predicted outcomes closer to desired flow | differentiable-FK / per-sample-backprop cost infeasible → keep V1/V2 result |

V3 is where differentiable FK and per-sample PointWorld backprop are tackled — **isolated
to a stage that only starts after V1 has proven the direction has value.** This is the
deliberate risk-reordering versus the original Plan A (which front-loaded all of it into
V1).

---

## 7. Relationship to the original "Plan A" (online differentiable reward)

Plan A = *policy action → differentiable FK → frozen PointWorld → grounding → reward,
backprop through PointWorld into the policy.* That is **V3 (inverse consistency)** here.
It was moved out of V1 because it bundled four simultaneous unknowns (differentiable FK
through urdfpy; ~1B-PTv3 per-sample backprop cost; a clean grounding objective; PointWorld
accuracy on LIBERO). The offline-distillation V1 falsifies the *last* unknown cheaply (C0
gate) and defers the first two to V3, keeping each stage a single variable — consistent
with the "probe-gate-then-train" methodology proven on the D4RT arm. Same endpoint,
better-ordered risk.

---

## 8. Scope boundaries (YAGNI)

**In scope (this spec):** C0 gate, C1–C5, V0, V1. V2/V3 sketched for continuity, specced
separately when reached.
**Explicitly deferred:** dense full-scene flow (§15.1 plan — sparse only); scheduled
teacher-forcing; candidate-action ranking (plan V4); geometry-backbone student (plan V5);
real-robot / LeRobot benchmark; online PointWorld at inference. These are **not** V1 work
and must not leak into the V1 implementation plan.

---

## 9. Evaluation

- **Primary:** LIBERO closed-loop task success rate, `pw_distill ∈ {off,on}`, byte-identical
  otherwise (LIBERO-Spatial/Object/Goal/Long).
- **Offline (diagnostic):** flow endpoint error, direction cosine, on a held-out split.
- **Robustness (V1 secondary claim):** camera-viewpoint and object-pose perturbation —
  the distillation's hypothesized benefit is spatial/physical robustness, so test it.
- **Ablation:** V0 · V1(aux-loss-only) · V1(flow tokens fed to action head).

---

## 10. Risks (beyond the C0 gate)

| Risk | Mitigation |
| --- | --- |
| Teacher flow wrong from bad depth | LIBERO sim depth is clean; confidence masks in cache weight the loss; C0 gate catches gross errors first. |
| Student can't infer metric 3D from RGB | dual views + temporal window + robot-base-frame targets; predict normalized flow if needed. |
| Flow head learns but doesn't help action | `feed_flow_tokens` couples flow into the head; ablate on/off; V3 inverse-consistency as the stronger coupling later. |
| Repo drift from starVLA conventions | enforced `[LangPointWorld]` marker + config-switch controlled experiment + reuse of host dataloader/eval — no standalone `project/` tree (rejecting source plan §14.1). |
| Coordinate-frame mismatch (teacher vs student) | pinned to robot-base frame in sim; asserted in cache-build and probe; single source of truth. |
