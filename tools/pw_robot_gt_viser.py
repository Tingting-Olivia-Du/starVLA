# [LangPointWorld] Diagnostic: overlay our robot_flows (magenta, PW's action input) against the
# REAL LIBERO gripper points from sim per-pixel segmentation (green, ground truth), in one viser,
# so we can SEE whether the Panda-FK gripper cloud aligns with the true gripper — no guessing off the
# official mesh. Frame slider animates both over time. Also shows the scene cloud (gray) for context.
import argparse, json, os, socket, sys, time
import numpy as np
import h5py

sys.path.insert(0, "/workspace/tingting/envs/pw_extra_site")
sys.path.insert(0, "/workspace/tingting/starVLA")
DATA = "/workspace/tingting/LIBERO/libero/datasets/libero_object"


def free_port(p):
    while True:
        s = socket.socket(); busy = s.connect_ex(("127.0.0.1", p)) == 0; s.close()
        if not busy:
            return p
        p += 1


def gt_gripper_points(z, name_by_bid, t):
    """Real gripper points at frame t from sim seg, in WORLD frame."""
    d = z["agentview_depth"][t][::-1].astype(np.float32); seg = z["agentview_seg_bodyid"][t]
    K = z["agentview_K"][t]; E = z["agentview_E_cam2world"][t]
    ys, xs = np.nonzero(np.isfinite(d) & (d > 1e-4) & (d < 5)); zz = d[ys, xs]
    pc = np.stack([(xs - K[0, 2]) / K[0, 0] * zz, (ys - K[1, 2]) / K[1, 1] * zz, zz], -1)
    Pw = (E @ np.concatenate([pc, np.ones((len(pc), 1))], -1).T).T[:, :3]
    gb = [bid for bid, nm in name_by_bid.items() if ("gripper" in nm or "finger" in nm or "hand" in nm)]
    return Pw[np.isin(seg[ys, xs], gb)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="/workspace/tingting/.tmp/s0_sim/alphabet_soup_demo_0.npz")
    ap.add_argument("--official", action="store_true", help="apply official preprocessing (centered frame)")
    ap.add_argument("--port", type=int, default=8099)
    args = ap.parse_args()

    from starVLA.model.modules.langpw.libero_to_datadict import LiberoDataDictBuilder
    base = os.path.basename(args.sim)[:-4]
    demo = base.split("_demo_")[1]; task = base.rsplit("_demo_", 1)[0]
    H = f"{DATA}/pick_up_the_{task}_and_place_it_in_the_basket_demo.hdf5"
    z = np.load(args.sim, allow_pickle=True)
    names = list(z["obj_names"]); op = z["obj_poses"]; base_T = op[0][names.index("robot0_base")]; base_inv = np.linalg.inv(base_T)
    b2n = json.loads(str(z["bodyid_to_name"])); name_by_bid = {int(k): v for k, v in b2n.items()}
    with h5py.File(H, "r") as f:
        o = f["data"][f"demo_{demo}"]["obs"]
        j = np.asarray(o["joint_states"][:], np.float64); g = np.asarray(o["gripper_states"][:], np.float64)

    b = LiberoDataDictBuilder(domain="droid", device="cuda", robot_urdf="panda")
    frame = "base" if args.official else "world"
    dd = b.build_from_sim(z, j, g, horizon=11, cams=("agentview",), frame=frame, official_preprocess=args.official)
    rf = dd["robot_flows"].astype(np.float32)                 # [T,Nr,3] our robot_flows (PW input)
    scene = dd["scene_flows"][0].astype(np.float32)
    shift = np.asarray(dd.get("__shift_amount__", np.zeros(3)), np.float32)
    ti = z["ti"]

    # GT gripper points per model-frame, transformed into the SAME frame as robot_flows
    gt = []
    for t in range(11):
        pw = gt_gripper_points(z, name_by_bid, t)   # sim npz depth/seg already the 11 sampled frames
        if args.official or frame == "base":
            pb = (base_inv[:3, :3] @ pw.T).T + base_inv[:3, 3] + shift
        else:
            pb = pw
        gt.append(pb.astype(np.float32))
    # per-frame centroid distance
    for t in [0, 5, 10]:
        print(f"[rgt] t={t}: robot centroid {np.round(rf[t].mean(0),3)} GT-gripper centroid {np.round(gt[t].mean(0),3)} "
              f"dist {np.linalg.norm(rf[t].mean(0)-gt[t].mean(0)):.3f}m", flush=True)

    # gripper_open [0,1] per frame (the FIXED feature): 1=open, 0=closed
    from starVLA.model.modules.langpw.libero_to_datadict import libero_gripper_open
    gopen = libero_gripper_open(g[ti])                      # (11,)
    print(f"[rgt] gripper_open (fixed [0,1]): {np.round(gopen,2)}  1=open 0=closed", flush=True)

    import viser
    p = free_port(args.port)
    server = viser.ViserServer(port=p); server.scene.set_up_direction("+z")
    gui_t = server.gui.add_slider("frame", 0, 10, 1, 0)
    gui_sc = server.gui.add_checkbox("scene (gray)", True)
    gui_go = server.gui.add_text("gripper_open (1=open,0=closed)", f"{gopen[0]:.2f}")

    def render():
        t = int(gui_t.value)
        gui_go.value = f"{gopen[t]:.2f}"                    # live gripper_open readout
        if gui_sc.value:
            server.scene.add_point_cloud("scene", scene, colors=(160, 160, 160), point_size=0.004)
        else:
            server.scene.remove_by_name("scene")
        # color the robot cloud by gripper_open: green(open) → red(closed), so you SEE the grasp close
        oc = gopen[t]
        rob_col = np.tile(np.array([[int(255 * (1 - oc)), int(255 * oc), 40]], np.uint8), (rf[t].shape[0], 1))
        server.scene.add_point_cloud("robot_flows", rf[t], colors=rob_col, point_size=0.010)  # green=open,red=closed
        server.scene.add_point_cloud("gt_gripper", gt[t], colors=(120, 120, 255), point_size=0.006)  # blue = GT gripper
    for gg in (gui_t, gui_sc):
        gg.on_update(lambda _: render())
    render()
    print(f"[rgt] viser READY  http://localhost:{p}  (ACTUAL {p}) | MAGENTA=our robot_flows, "
          f"GREEN=real LIBERO gripper (GT). They should overlap. frame={frame} official={args.official}", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
