# [LangPointWorld] Generate + visualize DUAL-VIEW GT scene flow (agentview + wrist) merged in the
# world frame, reusing the VERIFIED single-view logic from pw_make_finetune_hdf5.py:
#   per view: flip depth+rgb ([::-1]), NOT seg; unproject with real K/E; assign points to objects by
#   per-pixel seg; robust-radius outlier filter; rigid-body transform object points by the object's
#   GT pose trajectory. Merge both views' clouds. Object points get the SAME pose-traj transform
#   regardless of source view -> merged object motion is coherent.
# Purpose (user request): SEE whether dual-view merged object points are more complete, world-aligned,
# and conflict-free than single agentview — BEFORE deciding to re-pack/re-finetune.
import argparse, json, os, sys, time
import numpy as np

DATA = "/workspace/tingting/LIBERO/libero/datasets/libero_object"


def unproject(depth, rgb, K, E_c2w):
    H, W = depth.shape
    ys, xs = np.nonzero(np.isfinite(depth) & (depth > 1e-4) & (depth < 5))
    z = depth[ys, xs]; fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pc = np.stack([(xs - cx) / fx * z, (ys - cy) / fy * z, z], -1)
    Pw = (E_c2w @ np.concatenate([pc, np.ones((len(pc), 1))], -1).T).T[:, :3]
    return Pw.astype(np.float32), rgb[ys, xs], (ys, xs)


def build_view_flow(z, cam, T, op, name2idx, name_by_bid, max_pts):
    """One camera -> (Pw0[Ns,3], flows[T,Ns,3], colors[Ns,3], is_obj[Ns] bool). Same logic as packer."""
    depth = z[f"{cam}_depth"][0][::-1, :].astype(np.float32)     # opengl flip (depth+rgb), NOT seg
    rgb = z[f"{cam}_rgb"][0][::-1, :, :]
    seg = z[f"{cam}_seg_bodyid"][0]                              # aligned to UN-flipped image
    K = z[f"{cam}_K"][0]; E_c2w = z[f"{cam}_E_cam2world"][0]
    Pw0, cols, (ys, xs) = unproject(depth, rgb, K, E_c2w)
    body = seg[ys, xs]

    def role(nm):
        if nm.startswith(("robot0", "gripper", "mount")): return "robot"
        if nm in name2idx and nm not in ("floor", "world", "table"): return "obj"
        return "bg"
    roles = np.array([role(name_by_bid.get(int(b), "")) for b in body])
    keep = roles != "robot"
    Pw0, body, cols, roles = Pw0[keep], body[keep], cols[keep], roles[keep]

    # robust-radius outlier filter per object (same as packer)
    drop = np.zeros(len(Pw0), bool)
    for bid in np.unique(body):
        nm = name_by_bid.get(int(bid), "")
        if role(nm) != "obj": continue
        m = body == bid; ctr = op[0][name2idx[nm]][:3, 3]
        d = np.linalg.norm(Pw0[m] - ctr, axis=-1)
        med = np.median(d); mad = np.median(np.abs(d - med)) + 1e-6
        thr = max(0.10, med + 4.0 * mad)
        idx = np.where(m)[0]; drop[idx[d > thr]] = True
    keep2 = ~drop; Pw0, body, cols, roles = Pw0[keep2], body[keep2], cols[keep2], roles[keep2]

    # subsample keep all object points
    if len(Pw0) > max_pts:
        oi = np.where(roles == "obj")[0]; bi = np.where(roles != "obj")[0]
        rng = np.random.default_rng(0)
        sel = np.concatenate([oi, rng.choice(bi, max(0, max_pts - len(oi)), replace=False)])
        Pw0, body, cols, roles = Pw0[sel], body[sel], cols[sel], roles[sel]

    flows = np.tile(Pw0[None], (T, 1, 1)).astype(np.float32)
    for bid in np.unique(body):
        nm = name_by_bid.get(int(bid), "")
        if role(nm) != "obj": continue
        m = body == bid; oi = name2idx[nm]
        T0inv = np.linalg.inv(op[0][oi])
        Plocal = (T0inv[:3, :3] @ Pw0[m].T).T + T0inv[:3, 3]
        for t in range(T):
            Tt = op[t][oi]; flows[t][m] = (Tt[:3, :3] @ Plocal.T).T + Tt[:3, 3]
    is_obj = roles == "obj"
    return Pw0, flows, cols.astype(np.uint8), is_obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True)
    ap.add_argument("--cams", default="agentview,robot0_eye_in_hand")
    ap.add_argument("--max-pts-per-view", type=int, default=6000)
    ap.add_argument("--move-thresh", type=float, default=0.05)
    ap.add_argument("--viser", type=int, default=1)
    ap.add_argument("--port", type=int, default=8085)
    args = ap.parse_args()
    cams = tuple(args.cams.split(","))

    z = np.load(args.sim, allow_pickle=True)
    names = list(z["obj_names"]); name2idx = {n: i for i, n in enumerate(names)}
    b2n = json.loads(str(z["bodyid_to_name"])); name_by_bid = {int(k): v for k, v in b2n.items()}
    op = z["obj_poses"]; T = op.shape[0]

    per_view = {}
    for c in cams:
        Pw0, flows, cols, is_obj = build_view_flow(z, c, T, op, name2idx, name_by_bid, args.max_pts_per_view)
        per_view[c] = dict(Pw0=Pw0, flows=flows, cols=cols, is_obj=is_obj)
        disp = np.linalg.norm(flows[-1] - flows[0], axis=-1)
        print(f"[dualGT] {c}: {len(Pw0)} pts ({int(is_obj.sum())} obj), obj max disp {disp[is_obj].max() if is_obj.any() else 0:.3f}m, "
              f"movers(>{args.move_thresh}) {(disp > args.move_thresh).sum()}", flush=True)

    # merged
    flows_all = np.concatenate([per_view[c]["flows"] for c in cams], axis=1)   # [T, Ns_total, 3]
    cols_all = np.concatenate([per_view[c]["cols"] for c in cams], axis=0)
    view_id = np.concatenate([np.full(per_view[c]["Pw0"].shape[0], i) for i, c in enumerate(cams)])
    disp_all = np.linalg.norm(flows_all[-1] - flows_all[0], axis=-1)
    movers = disp_all > args.move_thresh
    print(f"[dualGT] MERGED {flows_all.shape[1]} pts, {int(movers.sum())} movers, max disp {disp_all.max():.3f}m", flush=True)

    # conflict check: do the two views' object points for the same object agree in world space?
    for c in cams:
        pv = per_view[c]; oc = pv["Pw0"][pv["is_obj"]]
        if len(oc):
            print(f"[dualGT] {c} obj-point centroid: {np.round(oc.mean(0),3)}", flush=True)

    if args.viser:
        import socket, viser
        # pick a truly-free port so requested == actual (viser silently hops to the next free port
        # when the requested one is busy, which is what caused the 8084-vs-8085 confusion).
        p = args.port
        while True:
            s = socket.socket(); busy = s.connect_ex(("127.0.0.1", p)) == 0; s.close()
            if not busy: break
            p += 1
        server = viser.ViserServer(port=p); server.scene.set_up_direction("+z")
        gui_t = server.gui.add_slider("frame", 0, T - 1, 1, T - 1)
        gui_hi = server.gui.add_checkbox("highlight movers (red)", True)
        gui_view = server.gui.add_checkbox("color by view (cam0=blue,cam1=green)", False)

        def render():
            t = int(gui_t.value); cols = cols_all.copy()
            if gui_view.value:
                cols[view_id == 0] = [80, 120, 255]; cols[view_id == 1] = [80, 255, 120]
            if gui_hi.value:
                cols[movers] = [255, 0, 0]
            server.scene.add_point_cloud("scene", flows_all[t].astype(np.float32),
                                         colors=cols.astype(np.uint8), point_size=0.004)
        for g in (gui_t, gui_hi, gui_view): g.on_update(lambda _: render())
        render()
        print(f"[dualGT] viser READY  http://localhost:{p}   (ACTUAL PORT {p}) | red=movers, "
              f"toggle 'color by view' to see cam0(blue)/cam1(green) coverage overlap", flush=True)
        while True:
            time.sleep(3600)


if __name__ == "__main__":
    main()
