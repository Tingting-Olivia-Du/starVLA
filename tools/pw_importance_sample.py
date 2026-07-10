# [LangPointWorld] Importance-aware (foveated) scene point sampling: keep the SAME total budget
# (~8192, in-distribution for PW) but REALLOCATE points — dense on the task object + gripper region,
# sparse on background/table/other objects. Uses the sim-render per-pixel segmentation to assign a
# role to every unprojected point, then samples per-role with a budget split. Viser compares
# UNIFORM vs IMPORTANCE sampling side by side (two point clouds, toggle) so we can SEE the effect
# on point distribution WITHOUT running PW.
import argparse, json, os, sys, socket, time
import numpy as np

DATA = "/workspace/tingting/LIBERO/libero/datasets/libero_object"


def free_port(p):
    while True:
        s = socket.socket(); busy = s.connect_ex(("127.0.0.1", p)) == 0; s.close()
        if not busy:
            return p
        p += 1


def unproject_with_roles(z, cam, name2idx, name_by_bid, op):
    """Return world points, colors, and a role array (0=task-obj,1=gripper,2=other-obj,3=bg)."""
    depth = z[f"{cam}_depth"][0][::-1].astype(np.float32)
    rgb = z[f"{cam}_rgb"][0][::-1]
    seg = z[f"{cam}_seg_bodyid"][0]                       # NOT flipped (aligned to un-flipped image)
    K = z[f"{cam}_K"][0]; E_c2w = z[f"{cam}_E_cam2world"][0]
    ys, xs = np.nonzero(np.isfinite(depth) & (depth > 1e-4) & (depth < 5))
    zz = depth[ys, xs]; fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pc = np.stack([(xs - cx) / fx * zz, (ys - cy) / fy * zz, zz], -1)
    Pw = (E_c2w @ np.concatenate([pc, np.ones((len(pc), 1))], -1).T).T[:, :3]
    cols = rgb[ys, xs]; body = seg[ys, xs]

    # the manipulated (task) object = max-displacement object
    cand = [n for n in name2idx if not n.startswith(("robot0", "gripper", "mount"))
            and n not in ("floor", "world", "table")]
    disps = {n: np.linalg.norm(op[-1][name2idx[n]][:3, 3] - op[0][name2idx[n]][:3, 3]) for n in cand}
    task_obj = max(disps, key=disps.get)

    def role(bid):
        nm = name_by_bid.get(int(bid), "")
        if nm == task_obj: return 0
        if nm.startswith(("robot0_gripper", "gripper")) or "finger" in nm or "hand" in nm: return 1
        if nm in name2idx and nm not in ("floor", "world", "table"): return 2
        return 3
    roles = np.array([role(b) for b in body], np.int8)
    return Pw.astype(np.float32), cols.astype(np.uint8), roles, task_obj


def sample_uniform(P, C, R, budget, rng):
    if len(P) <= budget:
        return P, C, R
    sel = rng.choice(len(P), budget, replace=False)
    return P[sel], C[sel], R[sel]


def sample_importance(P, C, R, budget, rng, split=(0.45, 0.15, 0.15, 0.25)):
    """Reallocate `budget` across roles by `split` (task-obj, gripper, other-obj, bg). Dense object."""
    out_idx = []
    alloc = [int(budget * s) for s in split]
    for r, k in enumerate(alloc):
        pool = np.where(R == r)[0]
        if len(pool) == 0:
            continue
        take = min(k, len(pool)) if r != 0 else min(max(k, len(pool)), len(pool))  # keep ALL task-obj if fewer
        take = min(take, len(pool))
        out_idx.append(rng.choice(pool, take, replace=False) if len(pool) > take else pool)
    idx = np.concatenate(out_idx) if out_idx else np.arange(min(budget, len(P)))
    # top up to budget from background if short
    if len(idx) < budget:
        rest = np.setdiff1d(np.arange(len(P)), idx)
        if len(rest):
            add = rng.choice(rest, min(budget - len(idx), len(rest)), replace=False)
            idx = np.concatenate([idx, add])
    return P[idx], C[idx], R[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="/workspace/tingting/.tmp/s0_sim/alphabet_soup_demo_0.npz")
    ap.add_argument("--cams", default="agentview,robot0_eye_in_hand")
    ap.add_argument("--budget", type=int, default=8192)
    ap.add_argument("--port", type=int, default=8092)
    args = ap.parse_args()
    cams = tuple(args.cams.split(","))

    z = np.load(args.sim, allow_pickle=True)
    names = list(z["obj_names"]); name2idx = {n: i for i, n in enumerate(names)}
    b2n = json.loads(str(z["bodyid_to_name"])); name_by_bid = {int(k): v for k, v in b2n.items()}
    op = z["obj_poses"]
    rng = np.random.default_rng(0)

    P, C, R = [], [], []
    task_obj = None
    for c in cams:
        p, col, r, task_obj = unproject_with_roles(z, c, name2idx, name_by_bid, op)
        # world bounds filter (same as build_from_sim)
        keep = (p[:, 0] > -1.5) & (p[:, 0] < 1.2) & (np.abs(p[:, 1]) < 1.2) & (p[:, 2] > -0.05) & (p[:, 2] < 1.5)
        P.append(p[keep]); C.append(col[keep]); R.append(r[keep])
    P = np.concatenate(P); C = np.concatenate(C); R = np.concatenate(R)

    per_budget = args.budget
    Pu, Cu, Ru = sample_uniform(P, C, R, per_budget, np.random.default_rng(0))
    Pi, Ci, Ri = sample_importance(P, C, R, per_budget, np.random.default_rng(0))

    def stats(tag, Rr):
        n = len(Rr)
        print(f"[imp] {tag:10s} total={n:5d} | task-obj={int((Rr==0).sum()):4d} gripper={int((Rr==1).sum()):3d} "
              f"other-obj={int((Rr==2).sum()):4d} bg={int((Rr==3).sum()):5d}", flush=True)
    print(f"[imp] task_obj={task_obj} | raw pool {len(P)} pts (task-obj {int((R==0).sum())})", flush=True)
    stats("UNIFORM", Ru); stats("IMPORTANCE", Ri)

    import viser
    p = free_port(args.port)
    server = viser.ViserServer(port=p); server.scene.set_up_direction("+z")
    mode = server.gui.add_dropdown("sampling", ["UNIFORM", "IMPORTANCE"], "IMPORTANCE")
    role_col = server.gui.add_checkbox("color by role (obj=red,grip=cyan,other=blue,bg=gray)", False)
    ROLE_RGB = np.array([[255, 40, 40], [0, 220, 220], [60, 100, 255], [150, 150, 150]], np.uint8)

    def render():
        Pp, Cc, Rr = (Pi, Ci, Ri) if mode.value == "IMPORTANCE" else (Pu, Cu, Ru)
        cc = ROLE_RGB[Rr] if role_col.value else Cc
        server.scene.add_point_cloud("cloud", Pp.astype(np.float32), colors=cc.astype(np.uint8), point_size=0.005)
    mode.on_update(lambda _: render()); role_col.on_update(lambda _: render()); render()
    print(f"[imp] viser READY  http://localhost:{p}   (ACTUAL PORT {p}) | dropdown=UNIFORM/IMPORTANCE, "
          f"toggle 'color by role' to see the reallocation", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
