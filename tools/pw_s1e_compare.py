# [LangPointWorld] S1-e: compare the 4 runs' val metrics (from each run's train_log.jsonl) and print
# the S1 claim table. Answers: (1) does flow-conditioning beat action-only (v1 vs v0)? (2) do flow
# tokens actually help actions, or decorative (v1 vs no-cond)? (3) does the finetuned teacher beat
# the un-finetuned base-PW teacher (v1 vs basePW)? Metric = best val_action (BC) + val_flow_EPE.
import glob, json, os

RUNS = {
    "v1_flow_cond":       "V1 (flow + condition_action) — MAIN",
    "v0_action_only":     "V0 (action-only baseline)",
    "v1_flow_nocond":     "no-cond (flow trained, NOT fed to action) — decorative control",
    "v1_basePW_teacher":  "basePW (flow from UN-finetuned teacher)",
}
ROOT = "/workspace/tingting/.tmp/s1_train"


def load(run):
    p = os.path.join(ROOT, run, "train_log.jsonl")
    if not os.path.exists(p):
        return None
    recs = [json.loads(l) for l in open(p)]
    return recs or None


def main():
    print(f"\n{'run':20s} {'steps':>6s} {'best_val_action':>16s} {'best_val_flow_EPE(m)':>20s} {'last_val_action':>16s}")
    print("-" * 88)
    summary = {}
    for run, desc in RUNS.items():
        recs = load(run)
        if not recs:
            print(f"{run:20s} {'--- no log yet ---'}")
            continue
        best_a = min(r["val_action"] for r in recs)
        best_epe = min(r.get("val_flow_epe_m", 9) for r in recs)
        last_a = recs[-1]["val_action"]
        summary[run] = (best_a, best_epe, last_a, recs[-1]["step"])
        print(f"{run:20s} {recs[-1]['step']:>6d} {best_a:>16.4f} {best_epe:>20.4f} {last_a:>16.4f}")
    print("-" * 88)
    print("\nCLAIMS:")
    NOISE = 0.02  # action FM-loss eval noise floor; gaps below this are NOT meaningful
    if "v1_flow_cond" in summary and "v0_action_only" in summary:
        v1, v0 = summary["v1_flow_cond"][0], summary["v0_action_only"][0]
        verdict = ("INCONCLUSIVE — gap within eval noise (no open-loop BC signal)" if abs(v1 - v0) < NOISE
                   else "V1 better" if v1 < v0 else "V0 better")
        print(f"  (1) V1 vs V0 action-loss: {v1:.4f} vs {v0:.4f} -> {verdict}")
    if "v1_flow_cond" in summary and "v1_flow_nocond" in summary:
        v1, nc = summary["v1_flow_cond"][0], summary["v1_flow_nocond"][0]
        verdict = ("INCONCLUSIVE — gap within eval noise (flow conditioning shows no open-loop benefit)"
                   if abs(v1 - nc) < NOISE else "flow tokens help" if v1 < nc else "no-cond better")
        print(f"  (2) V1 vs no-cond: {v1:.4f} vs {nc:.4f} -> {verdict}")
    if "v1_flow_cond" in summary and "v1_basePW_teacher" in summary:
        v1, bp = summary["v1_flow_cond"][1], summary["v1_basePW_teacher"][1]
        print(f"  (3) V1 vs basePW flow-EPE: {v1:.4f} vs {bp:.4f} -> "
              f"{'finetuned teacher gives better-distillable flow (S0 mattered)' if v1 < bp else 'base teacher comparable — investigate'}")
    print("\nNOTE: BC/flow-EPE are OPEN-LOOP proxies. Closed-loop LIBERO success (kami box) is the "
          "ultimate metric; export ckpts with pw_s1_export_ckpt.py.", flush=True)


if __name__ == "__main__":
    main()
