# [LangPointWorld] Viser for a PW PREDICTION npz (from pw_predict_density.py). Shows the model's
# IMAGINED scene flow animating over time (true color) + the GT manipulated-object trajectory
# (yellow line + moving dot) so you can SEE whether PW actually moves the grasped object. Also
# highlights the points PW predicts as "moving" (red) so its predicted motion is visible even when small.
import argparse, glob, os, sys, socket, time
import numpy as np


def free_port(p):
    while True:
        s = socket.socket(); busy = s.connect_ex(("127.0.0.1", p)) == 0; s.close()
        if not busy:
            return p
        p += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="a *_pred_<density>.npz from pw_predict_density.py")
    ap.add_argument("--move-thresh", type=float, default=0.03)
    ap.add_argument("--port", type=int, default=8091)
    args = ap.parse_args()

    z = np.load(args.pred, allow_pickle=True)
    sf = z["pred_scene_flows"].astype(np.float32)      # [T,Ns,3] PW-predicted
    cols = z["scene_colors"].astype(np.uint8)
    robot = z["robot_flows"].astype(np.float32)        # [T,Nr,3]
    obj = z["obj_traj_world"].astype(np.float32)       # [T,3] GT object center
    gt = float(z["gt_disp"]); dens = int(z["density"])
    T, Ns, _ = sf.shape
    disp = np.linalg.norm(sf[-1] - sf[0], axis=-1)     # per-point predicted displacement
    movers = disp > args.move_thresh
    pred_obj_p90 = float(np.percentile(disp[np.linalg.norm(sf[0] - obj[0], axis=-1) < 0.08], 90)) \
        if (np.linalg.norm(sf[0] - obj[0], axis=-1) < 0.08).any() else 0.0
    tag = os.path.basename(args.pred)[:-4]
    print(f"[predviz] {tag} density={dens} Ns={Ns} | GT obj disp {gt:.3f}m, PW near-obj p90 {pred_obj_p90:.3f}m "
          f"({100*pred_obj_p90/max(gt,1e-3):.0f}%) | {int(movers.sum())} predicted movers(>{args.move_thresh})", flush=True)

    import viser
    p = free_port(args.port)
    server = viser.ViserServer(port=p); server.scene.set_up_direction("+z")
    gui_t = server.gui.add_slider("frame", 0, T - 1, 1, T - 1)
    gui_hi = server.gui.add_checkbox("highlight PW-predicted movers (red)", True)
    gui_gt = server.gui.add_checkbox("GT object trajectory (yellow)", True)
    gui_rob = server.gui.add_checkbox("robot_flows (magenta)", True)

    def render():
        t = int(gui_t.value)
        c = cols.copy()
        if gui_hi.value:
            c[movers] = [255, 0, 0]
        server.scene.add_point_cloud("pred_scene", sf[t], colors=c, point_size=0.004)
        for nm in ("gt_traj", "gt_now", "robot"):
            server.scene.remove_by_name(nm)
        if gui_gt.value:
            if t > 0:
                segs = np.stack([obj[:t], obj[1:t + 1]], 1).astype(np.float32)
                server.scene.add_line_segments("gt_traj", segs, colors=(255, 255, 0), line_width=5.0)
            server.scene.add_icosphere("gt_now", radius=0.02, color=(255, 200, 0), position=obj[t])
        if gui_rob.value:
            ti = min(t, robot.shape[0] - 1)
            server.scene.add_point_cloud("robot", robot[ti], colors=(255, 0, 255), point_size=0.006)
    for g in (gui_t, gui_hi, gui_gt, gui_rob):
        g.on_update(lambda _: render())
    render()
    print(f"[predviz] viser READY  http://localhost:{p}   (ACTUAL PORT {p}) | red=PW-predicted movers, "
          f"yellow=GT object path, magenta=robot. Slider animates PW's prediction.", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
