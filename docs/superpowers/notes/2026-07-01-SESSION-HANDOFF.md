# PointWorld Distillation — Overnight Session Handoff (2026-07-01)

**For: your return. TL;DR — the entire teacher→student infrastructure is built, tested, and runs
end-to-end on real LIBERO data. The PW-PROBE gate came back BORDERLINE (corr 0.48; 3/8 episodes
track strongly at 0.8–0.99). I did NOT start V0/V1 training, honoring the "don't train on an
unclear gate" kill-criterion + your "no speculative GPU-hours" constraint. Everything is staged to
train the moment you decide the gate is cleared.**

Branch: `starVLA_dev`. All work marked `[LangPointWorld]`, committed. Nothing on GPU 0/1/4-7 was
touched; only GPU 2 used; only my own processes; only `/workspace/tingting/` files.

## What I built and verified (all green)

| Piece | Status | Evidence |
|---|---|---|
| Design spec + 2 impl plans (probe-gate, V1) | ✅ | `docs/superpowers/specs/2026-07-01-*`, `docs/superpowers/plans/2026-07-01-*` |
| Env (isolated `pw_extra_site`, no shared-env mutation) | ✅ | full PointWorld stack imports; 3 ckpts downloaded; DINOv3 code+weights resolved |
| VGGT geometry provider (depth+K+E from LIBERO RGB) | ✅ | verified: depth 128², fx≈207, cx=64 |
| LIBERO→PointWorld data_dict bridge | ✅ | scene_flows via VGGT-depth unproject + robot_flows via FK + real 31/16-dim features |
| Frozen teacher wrapper (large-droid 1288M) | ✅ | loads; auto-detects multi-domain ckpts |
| **Full imagine() end-to-end** | ✅ | LIBERO→VGGT→bridge→teacher → `scene_flows (11,8192,3)` finite |
| V1 modules (flow head, distill loss, sampling, ee-frame, metrics) | ✅ | 18 unit tests pass |
| PW-PROBE gate (8 episodes, corr/L2 in EE frame) + viz | ✅ ran | verdict below |

## The gate result (the decision point)

**large-droid, corrected motion-mode: corr_mean = 0.48 — BORDERLINE.** Per-episode:
ep0=0.99, ep6=0.85, ep7=0.81 (strong, pred/GT magnitudes matched), ep1/ep4≈0.37, ep2/ep3/ep5≈0/neg.
→ The teacher captures real LIBERO dynamics on ~half the episodes; partial transfer, **not** a dead
or under-driving teacher. Full analysis: `docs/superpowers/notes/2026-07-01-pw-probe-verdict.md`.

Two caveats that make me read 0.48 as an *under-estimate* of the teacher's real value:
1. **The GT is EE-position motion; the teacher predicts SCENE motion.** Scene flow only moves where
   the arm pushes objects, after contact — so an episode can score low even with a fine scene
   prediction. The rigorous fix is an object-pose GT probe.
2. **The true arbiter is closed-loop success-rate delta, not this proxy corr.** The gate is a cheap
   risk-reducer, not proof.

## Why I stopped here (and didn't train)

A borderline gate is not a clean pass. Training V0/V1 (~hours of GPU) on a teacher that may be
supplying noisy labels on half the episodes is exactly the speculative burn you told me to avoid,
and the spec's kill-criterion says don't train on a gate that didn't clearly pass. So I banked all
the infrastructure and stopped at the decision boundary — that's yours to make.

## Your options on return (ranked by my recommendation)

1. **Rigorous: object-pose GT probe.** Decode per-object 3D motion from LIBERO sim (the 110-dim
   MuJoCo `states`) in the `libero-ttd`/robosuite env (NOT in the starVLA env — this is why I
   couldn't do it autonomously). Re-probe teacher scene-flow vs object motion. If corr clears 0.7
   on the strong episodes, the gate passes cleanly → train.
2. **Pragmatic pilot: just train V1 vs V0 on `libero_object` and let closed-loop decide.** The 3
   strong episodes show real signal; confidence-weight the distill loss so noisy samples matter
   less. If V1 > V0 success rate, the gate proxy was simply too strict. Cache-build + train are
   fully coded in the V1 plan; I can kick this off in ~20 min of wiring when you say go.
3. **Fix the 42-dim BEHAVIOR feature layout** and probe `large-droid+behavior` (sim-trained, may
   transfer to LIBERO-sim better).

Tell me which and I'll execute. My lean: do (2) as a fast empirical read while (1) is set up — the
closed-loop delta is the number that actually matters.

## Repro / entry points
- Probe: `CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/workspace/tingting/envs/pw_extra_site:/workspace/tingting/starVLA HF_HOME=/root/angli/hf_cache TORCH_HOME=/root/angli/torch_cache python tools/pw_probe_libero.py --ckpt <large-droid> --libero-glob '<...>/libero_object/*demo.hdf5' --episodes 8 --out-dir results/pw_probe_motion`
- Viz: `python tools/pw_probe_visualize.py --probe-dir results/pw_probe_motion` → PNGs in `viz/`
- Imagine smoke: `python tools/_pw_imagine_smoke.py <HDF5> <CKPT>`
- Memory: `[[lang-pointworld-distill-design]]` has the full technical trail.
