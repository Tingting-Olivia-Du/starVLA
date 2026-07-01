# [LangPointWorld] PW-PROBE visualization (spec §4 metric 3): per-episode imagined vs GT
# EE-frame flow, saved as PNGs (headless-friendly for autonomous runs; no viser server).
# Green = predicted (teacher), Red = GT (EE trajectory), both in the current-frame EE frame.
# Part of the PointWorld point-flow distillation arm — see
# docs/superpowers/specs/2026-07-01-lang-pointworld-distill-design.md
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", default="/workspace/tingting/starVLA/results/pw_probe")
    args = ap.parse_args()
    eps = sorted(glob.glob(os.path.join(args.probe_dir, "ep*.npz")))
    if not eps:
        print(f"[viz] no ep*.npz in {args.probe_dir}")
        return
    out_dir = os.path.join(args.probe_dir, "viz")
    os.makedirs(out_dir, exist_ok=True)
    for ep in eps:
        d = np.load(ep, allow_pickle=True)
        pred, gt = d["pred_flow_ee"], d["gt_flow_ee"]     # [H-1,3] cumulative displacement/step
        # cumulative trajectory from EE origin
        origin = np.zeros(3)
        pred_traj = np.vstack([origin, pred])
        gt_traj = np.vstack([origin, gt])
        fig = plt.figure(figsize=(12, 4))
        for i, (a, b, title) in enumerate([(0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ")]):
            ax = fig.add_subplot(1, 3, i + 1)
            ax.plot(pred_traj[:, a], pred_traj[:, b], "g.-", label="pred (teacher)")
            ax.plot(gt_traj[:, a], gt_traj[:, b], "r.-", label="GT (EE)")
            ax.set_title(f"{os.path.basename(ep)} [{title}]")
            ax.set_xlabel("xyz"[a]); ax.set_ylabel("xyz"[b])
            ax.legend(fontsize=7); ax.axis("equal")
        png = os.path.join(out_dir, os.path.basename(ep).replace(".npz", ".png"))
        fig.tight_layout(); fig.savefig(png, dpi=90); plt.close(fig)
        print(f"[viz] {png}")
    print(f"[viz] wrote {len(eps)} plots to {out_dir}")


if __name__ == "__main__":
    main()
