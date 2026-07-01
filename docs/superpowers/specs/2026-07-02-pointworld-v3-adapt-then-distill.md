# Sample-Efficient Adaptation of a Foundation 3D World Model, then Distillation into an RGB-Only Language Policy

**v3 — grounded in measured evidence. Supersedes v2's assumption that the foundation teacher is usable zero-shot.**

Working title: *Adapt-then-Distill: Sample-Efficient Domain Adaptation of a Foundation 3D World Model for RGB-Only Language-Guided Manipulation*

---

## 0. What changed from v2, and why (the evidence)

v2 assumed PointWorld could be used as a **zero-shot** privileged teacher and made Experiment 0 (teacher-fidelity gate) a to-do. **We ran that gate. It is RED on LIBERO — but in a specific, adaptable way.** v3 turns that finding into the contribution.

**Measured facts (this repo, `[LangPointWorld]` tooling):**

| Probe | Result | Source |
|---|---|---|
| PW on **official BEHAVIOR** WDS | l2 = **0.00106 m** (mm-level), conf 0.9999 | official `eval.py`, `pw_viz_official_data.py` |
| PW on **LIBERO** (aligned, sim depth/K/E) | scene l2 ~0.25; **predicts only 5.1% of grasped-object motion** over 6 episodes | `pw_batch_grasp_check.py` |
| PW capability profile on LIBERO | static bg ✓ (~0.01); **object missed** (ratio 0.00–0.07); **arm region MIXED**: fires but under-drives (10–50% of GT) and direction unstable (cos +0.90/+0.62 but also −0.52/−0.28/−0.17) | `pw_capability_profile.py` |

**Interpretation — the dynamics *skeleton* is intact, the domain *calibration* is not.** PW is a near-perfect action-conditioned 3D dynamics model in-domain; on LIBERO it still fires near the arm and correctly leaves the background static, but (a) mis-maps LIBERO arm kinematics (direction+magnitude) and (b) lacks the grasp/contact→object-follows causality. This is a **domain-adaptation problem, not a rebuild** — which is exactly what makes *sample-efficient* adaptation plausible and interesting.

**Thesis shift:** the deployable contribution is no longer "distill a zero-shot foundation WM." It is:

> **Adapt-then-Distill:** with a *small* amount of target-domain data, cheaply adapt a foundation 3D world model so its privileged dynamics (incl. grasp causality) transfer to the new embodiment/scene, **then** distill that adapted teacher into an RGB-only language policy via flow distillation + world-model-in-the-loop counterfactual supervision. At deployment: dual-view RGB + language + proprio, no depth, no world model.

Novelty framing: **sample-efficient domain adaptation of a foundation dynamics model** (skeleton-intact → calibration-only → few samples) is the headline, and it is *supported by our profiling data*, not asserted.

---

## 1. Positioning (the narrow gap, updated)

v2's comparison table stands; add one row that is now ours alone:

| Axis | v2's neighbors (RoboFlow4D / FLIP / μ0 / RMA) | **Ours (v3)** |
|---|---|---|
| Teacher | from-scratch flow WM / test-time planner / low-dim oracle | **foundation 3D WM, SAMPLE-EFFICIENTLY ADAPTED to target** |
| Adaptation evidence | — | **domain-gap profiling → calibration-not-rebuild → few-shot adapt** |
| Counterfactual supervision | mostly no | **yes (inverse consistency + candidate ranking)** |
| Deploy input | RGB(-D) | **RGB + lang + proprio, no WM at test** |

Two differentiators survive review even after the finetune concession:
1. **We never train a dynamics model from scratch** — we *adapt* a foundation one with little data, inheriting its in-the-wild prior. RoboFlow4D trains its own.
2. **The adaptation is the contribution, quantified.** We show *how much* target data flips PW from "misses 95% of object motion" to usable, and that a skeleton-intact model needs far less than training from scratch.

If a reviewer says "you just finetuned + distilled," the counter is: (a) the profiling shows what specifically transfers vs not (an analysis result), (b) the sample-efficiency curve (how little data suffices), (c) the counterfactual coupling (S2/S3) no neighbor has.

---

## 2. Pipeline overview (stages)

```
S0  ADAPT  : finetune PW on a SMALL LIBERO set (per-object GT flow from sim) → PW_libero
Exp0*      : re-run the teacher-fidelity profile on PW_libero (was RED; must go GREEN to proceed)
S1  DISTILL: demo-action 3D-flow distillation from PW_libero into the RGB-only student
S2  COUPLE : world-model-in-the-loop inverse consistency (stop-gradient relabel with frozen PW_libero)
S3  RANK   : counterfactual action ranking (ablation tier)
Deploy     : student only — dual-view RGB + lang + proprio, no depth, no WM
```

**Everything downstream of S0 is v2's design, reused.** v3's new, load-bearing piece is **S0 + Exp0\*** — the adapt step and its gate.

---

## 3. S0 — Sample-efficient adaptation of PointWorld to LIBERO (the new core)

### 3.1 Data generation (already prototyped, `[LangPointWorld]`)
LIBERO gives us everything to build PW's training target **without touching mesh assets**:
- `tools/pw_sim_render.py` (libero env): replay HDF5 states → **real metric depth + real K/E + per-object 6D pose trajectories** (verified: scene/robot/objects share one metric frame after the base-frame fix).
- **Per-object GT scene flow** = assign each sim-depth surface point to its nearest object (by t=0 proximity to that object's pose), then transform by the object's GT pose trajectory across the 11-frame window. This reproduces the official `local_scene_points × scene_mesh_trajectories` supervision that PW was trained on — but sourced from sim depth + poses, not mesh sampling.
- Robot action = FK point-flow (already built), base-frame aligned.
- Package to HDF5 → WDS via the `data`-branch `data_integrity_check.py → make_wds_manifest.py → convert_wds.py` (validated on BEHAVIOR; `/workspace/tingting/PointWorld-data`).

### 3.2 Finetune
`train.py --model_path <large-droid+behavior ckpt> --domains libero --data_dirs <libero_wds> ...`
(PW supports finetune-from-checkpoint: `trainer.load_checkpoint_from_path` restores weights.) Freeze
the DINOv3 scene encoder; adapt the dynamics predictor + robot/scene projections. Start with a
**small budget** (e.g. libero_object 500 demos ≈ 5000 windows) and grow only if Exp0* demands.

### 3.3 The sample-efficiency study = a contribution, not a footnote
Sweep the adaptation set size (e.g. 1, 2, 5, 10 tasks; 10%–100% of demos) and plot **teacher fidelity
vs. #target samples**: grasped-object-motion ratio, arm-direction cosine, static-bg error. The
skeleton-intact hypothesis predicts a **steep early gain** (few samples fix calibration). This curve
*is* the paper's central quantitative claim about "sample-efficient adaptation of a foundation WM."

### 3.4 Exp0\* gate (re-run, must pass before S1)
Re-run `pw_capability_profile.py` + `pw_batch_grasp_check.py` on PW_libero. **Gate: grasped-object
ratio ≫ current 0.05 (target ≥ ~0.6) and arm-direction cosine consistently > 0 across episodes.** If
S0 fails to move these even at full budget, PW's architecture cannot fit LIBERO grasp causality and
we pivot honestly (§8). If it passes, distillation now supervises signal, not noise.

---

## 4. Student (RGB-only, deployable) — unchanged from v2

Dual-view RGB temporal window + token-level language + proprio → language-conditioned 4D query
transformer → {desired point-flow head `F_student∈R^{N×H_f×3}`, flow-matching action head
`A_student` conditioned on flow tokens}. RGB encoder v1 = frozen SigLIP/DINOv2 + temporal-view
transformer. **Flow tokens MUST feed the action head** (hard requirement + ablation). Scales:
K=2–4, H_a=4–16, H_f=4–8, N=128–256. `act2ptflow` (URDF→PW robot-point-flow action encoding) is
already built and base-frame validated; reuse for S2/S3.

---

## 5. Supervision signals (v2's S1/S2/S3, now fed by PW_libero)

- **S1 demo-flow distillation:** `L_flow = Σ w_i ρ(F_student − F_teacher)`, `w_i ∝ conf·[vis]`,
  up-weight object/gripper/contact/high-motion. Optional `L_direction`, `L_vis`. *Now teacher flow
  actually contains the object motion* (post-S0), so S1 no longer teaches "object stays put."
- **S2 inverse consistency:** periodic **stop-gradient** relabel — roll `A_student` through frozen
  PW_libero, use `F_rollout` as target for the desired-flow head + consistency via flow tokens. No
  backprop through / no surrogate of the WM (keeps "we never train a dynamics model from scratch").
- **S3 counterfactual ranking (ablation tier):** M∈4–8 candidates on a state subset; task-templated
  `Score(F_m, language, obj/target pts)`; `L_rank = CE(softmax(s/τ), p_student)`.

Staged objective: `L = L_action + λ1 L_flow (+λ2 L_inv +λ3 L_rank)`; add each only after it earns its keep.

---

## 6. Coordinate frames (SOLVED here, keep as method detail)

This was v2's "central feasibility risk." **We solved the teacher-side frame problem:** sim depth is
metric and in the robot-base world frame; scene+robot+objects align with zero error after the
base-frame fix (robot base at world (-0.6,0,0); cam extrinsic = E_w2c @ base_T). See
`docs/superpowers/notes/2026-07-02-libero-pointworld-alignment.md`. The remaining frame question is
**student-side**: can an RGB-only student emit metric 3D? Keep v2's **metric vs normalized vs
object-relative** ablation as a genuine result; default object-relative + base-frame hybrid. Point
selection: 96 object / 48 target / 48 gripper-contact / 64 background (N=256).

---

## 7. Experiments

- **Exp0\* (gate, §3.4):** teacher fidelity on PW_libero + the sample-efficiency curve (§3.3). This
  is now a headline figure, not just a go/no-go.
- **Main:** S1 improves RGB-only policy (robustness-primary — π₀/GR00T near ceiling on raw LIBERO
  success); S2 lowers ‖F_rollout−F_desired‖ + raises closed-loop success; S3 beats BC on
  recovery/compounding-error.
- **Robustness (primary axis):** camera-viewpoint · object-pose OOD · novel instances · occlusion ·
  lighting/texture · LIBERO-Long.
- **Ablations:** V0 / V0+flow(no teacher) / V1=+S1 / V2=+lang / V3=+S2 / V4=+S3 / V5=+geom; plus the
  **no-flow-conditioning control** (is flow decorative?) and the **adaptation-budget sweep**.
- **Benchmarks:** LIBERO → RoboMimic → real/LeRobot RGB-only deploy.

---

## 8. Scope & honest failure modes

**Paper = S0 (adapt + sample-efficiency curve) + Exp0\* + S1 + S2 + robustness.** Tight story:
*sample-efficiently adapt a foundation 3D WM to a new domain, then distill with world-model-in-loop
consistency into an RGB-only policy.* S3 ranking and V5 geometry backbone = ablation-tier/future.

**Kill criteria (honest):**
- If S0 can't lift grasped-object ratio even at full LIBERO budget → PW can't fit LIBERO grasp
  causality; pivot to the analysis paper ("what a foundation 3D WM does/doesn't transfer, and why").
- If post-S0 distillation gives no robustness gain over V0 → report negative; the sample-efficiency
  + domain-gap analysis is still publishable.
- If metric-3D is unlearnable RGB-only and object-relative gives no gain → soften "3D" claims.

---

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| S0 doesn't learn grasp causality | budget sweep; freeze DINOv3, adapt dynamics head; if flat → pivot (§8) |
| Sample-efficiency claim weak (needs lots of data) | that itself is the measured result; frame honestly (adaptation cost curve) |
| Per-object GT flow from depth is noisy | sim depth is clean+metric; confidence-weight; validate vs official mesh-based flow on a check set |
| act2ptflow mismatch | already base-frame validated against sim FK; re-validate on adapted PW |
| Reviewer: "just finetune+distill" | profiling result + sample-efficiency curve + counterfactual coupling (S2/S3) + zero-test-time-WM deploy |
| Baseline near ceiling | robustness/OOD primary, LIBERO-Long |

---

## 10. Timeline

1. **Wk 1–2:** per-object GT flow generator (sim-render → assign-to-object → pose-transform → HDF5→WDS); validate against official mesh-flow format.
2. **Wk 3–4:** **S0 finetune PW on libero_object** + Exp0\* re-profile; **sample-efficiency sweep** (the key early result / go-no-go).
3. **Wk 5–6:** V0 student baseline.
4. **Wk 7–9:** V1 (S1 distill from PW_libero) + scheduled conditioning + frame ablation.
5. **Wk 10–12:** V2 (lang grounding) + V3 (S2 inverse consistency) + robustness suite.
6. **Wk 13+:** RoboMimic ablations; RoboFlow4D-style head-to-head; (budget) S3; real RGB-only deploy; write-up.

---

## 11. What's already done (de-risks the plan)

From the `[LangPointWorld]` work in `/workspace/tingting/starVLA`:
- ✅ sim-render (real depth/K/E/object-poses), base-frame alignment (robot stands in scene, 0.03 contact).
- ✅ `build_from_sim` multi-view (agentview+wrist) data_dict; teacher loads + imagines end-to-end.
- ✅ Exp0 evidence: PW mm-accurate on BEHAVIOR, misses 95% object motion on LIBERO, capability profile.
- ✅ official data download + WDS + official-viz driving; per-object flow rendering.
- ✅ act2ptflow (FK robot point-flow), base-frame validated.
- ⬜ TODO: per-object GT flow generator → LIBERO WDS → **S0 finetune** → Exp0\* → student.

**The single highest-value next action is S0 §3.1–3.4:** generate the LIBERO per-object GT flow set,
finetune PW, and produce the sample-efficiency curve — it converts the (currently RED) teacher gate
into the paper's headline contribution, or tells us to pivot, cheaply.
