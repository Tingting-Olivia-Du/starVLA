# [LangPointWorld] Diagnostic: feed LIBERO dual-view (agentview + eye_in_hand) images to VGGT and
# SAVE the depth maps so we can eyeball whether VGGT depth is the reason PointWorld can't "see" the
# object. For each requested (task, demo): runs VGGT dual_view_t0, saves the input RGBs + VGGT depth
# (colormapped) and — when a sim-render npz exists — the REAL robosuite GT depth side-by-side, plus
# a quantitative comparison (range, per-pixel error vs GT after scale-align). VGGT depth is UP-TO-SCALE,
# so we also report the best-fit scale to GT (a huge scale/shape mismatch => VGGT depth is the culprit).
#
# Run in starVLA env (has vendored vggt). Example:
#   CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/workspace/tingting/starVLA \
#   python tools/pw_vggt_depth_check.py --task alphabet_soup --demos 0,5,12 \
#     --out-dir /workspace/tingting/.tmp/vggt_depth_check
import argparse, os, sys, glob
import numpy as np
import cv2

sys.path.insert(0, "/workspace/tingting/starVLA")

DATA = "/workspace/tingting/LIBERO/libero/datasets/libero_object"
SIM_DIR = "/workspace/tingting/.tmp/s0_sim"   # sim-render npz (real GT depth), if present


def colormap_depth(d, valid=None, vmin=None, vmax=None):
    """float depth [H,W] -> BGR colormap. Invalid/zero -> black."""
    d = d.astype(np.float32).copy()
    m = np.isfinite(d) & (d > 1e-6) if valid is None else valid
    if vmin is None:
        vmin = np.percentile(d[m], 2) if m.any() else 0.0
    if vmax is None:
        vmax = np.percentile(d[m], 98) if m.any() else 1.0
    n = np.clip((d - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    u8 = (n * 255).astype(np.uint8)
    col = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    col[~m] = 0
    return col, float(vmin), float(vmax)


def load_gt_depth(task, demo, cam="agentview"):
    """Real robosuite metric depth from sim-render npz (t=0), if available. Returns (depth, rgb) or None.
    NOTE sim-render depth is opengl-flipped vs the seg/rgb the packer uses; here we return the raw t=0
    depth + raw t=0 rgb for a like-for-like VGGT-vs-GT comparison (both un-flipped native)."""
    p = os.path.join(SIM_DIR, f"{task}_demo_{demo}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    d = z[f"{cam}_depth"][0].astype(np.float32)         # [H,W] real metric meters
    rgb = z[f"{cam}_rgb"][0]
    return d, rgb


def best_scale(vggt_d, gt_d, valid):
    """VGGT depth is up-to-scale; find s minimizing ||s*vggt - gt|| on valid px. Report s + rel error."""
    v = vggt_d[valid].ravel(); g = gt_d[valid].ravel()
    m = np.isfinite(v) & np.isfinite(g) & (v > 1e-6) & (g > 1e-6)
    if m.sum() < 50:
        return None
    v, g = v[m], g[m]
    s = float((v * g).sum() / (v * v).sum())            # least-squares scale
    rel = float(np.abs(s * v - g).mean() / (g.mean() + 1e-6))
    corr = float(np.corrcoef(v, g)[0, 1])               # shape agreement (scale-invariant)
    return {"scale": s, "rel_err": rel, "corr": corr, "n": int(m.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="alphabet_soup")
    ap.add_argument("--demos", default="0,5,12", help="comma list of demo indices")
    ap.add_argument("--out-dir", default="/workspace/tingting/.tmp/vggt_depth_check")
    ap.add_argument("--cams", default="agentview_rgb,eye_in_hand_rgb")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cam_keys = tuple(args.cams.split(","))
    demos = [d.strip() for d in args.demos.split(",")]

    from starVLA.model.modules.langpw.vggt_geometry import VGGTGeometryProvider
    geo = VGGTGeometryProvider(device="cuda")
    H = f"{DATA}/pick_up_the_{args.task}_and_place_it_in_the_basket_demo.hdf5"
    assert os.path.exists(H), f"missing hdf5 {H}"
    print(f"[vggt-check] task={args.task} demos={demos} cams={cam_keys}", flush=True)

    for demo in demos:
        out = geo.dual_view_t0(H, f"demo_{demo}", cam_keys=cam_keys, t=0)
        panels = []
        for ck in cam_keys:
            g = out[ck]
            rgb = g["rgb"]; d = g["depth"]                      # native res
            dcol, vmin, vmax = colormap_depth(d)
            rgb_bgr = rgb[:, :, ::-1]
            # upscale small (128) panels for visibility
            up = 3
            rgb_u = cv2.resize(rgb_bgr, None, fx=up, fy=up, interpolation=cv2.INTER_NEAREST)
            d_u = cv2.resize(dcol, None, fx=up, fy=up, interpolation=cv2.INTER_NEAREST)
            cv2.putText(rgb_u, f"{ck} RGB", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(d_u, f"VGGT depth [{vmin:.2f},{vmax:.2f}]", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            row = [rgb_u, d_u]

            # GT comparison for agentview (sim-render only rendered agentview+wrist; agentview GT is metric)
            gt = load_gt_depth(args.task, demo, cam="agentview") if ck == "agentview_rgb" else None
            if gt is not None:
                gt_d, gt_rgb = gt
                gtcol, gvmin, gvmax = colormap_depth(gt_d)
                gt_u = cv2.resize(gtcol, None, fx=up, fy=up, interpolation=cv2.INTER_NEAREST)
                cv2.putText(gt_u, f"GT depth [{gvmin:.2f},{gvmax:.2f}]m", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                row.append(gt_u)
                valid = np.isfinite(gt_d) & (gt_d > 1e-4) & (gt_d < 5)
                cmp = best_scale(d, gt_d, valid)
                if cmp:
                    print(f"[vggt-check] demo_{demo} {ck}: VGGT-vs-GT scale={cmp['scale']:.3f} "
                          f"rel_err={cmp['rel_err']:.2f} shape-corr={cmp['corr']:.3f} (n={cmp['n']})", flush=True)
            panels.append(np.hstack(row))
        # pad rows to same width then stack
        W = max(p.shape[1] for p in panels)
        panels = [cv2.copyMakeBorder(p, 0, 0, 0, W - p.shape[1], cv2.BORDER_CONSTANT, value=0) for p in panels]
        fig = np.vstack(panels)
        outp = os.path.join(args.out_dir, f"{args.task}_demo_{demo}_vggt_depth.png")
        cv2.imwrite(outp, fig)
        # also save raw VGGT depth npy for numeric inspection
        np.savez(os.path.join(args.out_dir, f"{args.task}_demo_{demo}_depth.npz"),
                 **{ck.replace("_rgb", ""): out[ck]["depth"] for ck in cam_keys},
                 **{ck.replace("_rgb", "") + "_K": out[ck]["K"] for ck in cam_keys})
        print(f"[vggt-check] wrote {outp}", flush=True)
    print(f"[vggt-check] DONE -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
