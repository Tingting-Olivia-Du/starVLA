# [LangPointWorld] Compare TWO PW prediction npzs in one viser (e.g. uniform vs importance sampling).
# Dropdown switches between them; shows predicted scene flow (true color) + GT object trajectory
# (yellow) + optional PW-predicted-mover highlight (red). Lets you SEE if importance sampling changed
# what PW predicts for the grasped object.
import argparse, glob, os, socket, time
import numpy as np


def free_port(p):
    while True:
        s = socket.socket(); busy = s.connect_ex(("127.0.0.1", p)) == 0; s.close()
        if not busy:
            return p
        p += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="/workspace/tingting/.tmp/pw_pred_imp/*_pred_*.npz")
    ap.add_argument("--move-thresh", type=float, default=0.03)
    ap.add_argument("--port", type=int, default=8093)
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    assert files, f"no npz matched {args.glob}"
    data = {}
    labels = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        lab = os.path.basename(f)[:-4].split("_pred_")[-1]   # e.g. "uniform_8192"
        data[lab] = dict(sf=z["pred_scene_flows"].astype(np.float32), cols=z["scene_colors"].astype(np.uint8),
                         robot=z["robot_flows"].astype(np.float32), obj=z["obj_traj_world"].astype(np.float32),
                         gt=float(z["gt_disp"]))
        labels.append(lab)
        d = data[lab]; disp = np.linalg.norm(d["sf"][-1] - d["sf"][0], axis=-1)
        near = np.linalg.norm(d["sf"][0] - d["obj"][0], axis=-1) < 0.08
        p90 = float(np.percentile(disp[near], 90)) if near.any() else 0
        print(f"[cmp] {lab:20s} Ns={d['sf'].shape[1]:5d} GT={d['gt']:.3f}m PW-near-obj-p90={p90:.3f}m ({100*p90/max(d['gt'],1e-3):.0f}%)", flush=True)

    import viser
    p = free_port(args.port)
    server = viser.ViserServer(port=p); server.scene.set_up_direction("+z")
    sel = server.gui.add_dropdown("prediction", labels, labels[0])
    gui_t = server.gui.add_slider("frame", 0, data[labels[0]]["sf"].shape[0] - 1, 1, data[labels[0]]["sf"].shape[0] - 1)
    gui_hi = server.gui.add_checkbox("highlight PW movers (red)", True)
    gui_gt = server.gui.add_checkbox("GT object path (yellow)", True)

    def render():
        d = data[sel.value]; sf = d["sf"]; t = int(gui_t.value)
        disp = np.linalg.norm(sf[-1] - sf[0], axis=-1); movers = disp > args.move_thresh
        c = d["cols"].copy()
        if gui_hi.value:
            c[movers] = [255, 0, 0]
        server.scene.add_point_cloud("pred", sf[min(t, sf.shape[0]-1)], colors=c, point_size=0.004)
        for nm in ("gt_traj", "gt_now"):
            server.scene.remove_by_name(nm)
        if gui_gt.value:
            obj = d["obj"]
            if t > 0:
                server.scene.add_line_segments("gt_traj", np.stack([obj[:t], obj[1:t+1]], 1).astype(np.float32),
                                               colors=(255, 255, 0), line_width=5.0)
            server.scene.add_icosphere("gt_now", radius=0.02, color=(255, 200, 0), position=obj[min(t, len(obj)-1)])
    for g in (sel, gui_t, gui_hi, gui_gt):
        g.on_update(lambda _: render())
    render()
    print(f"[cmp] viser READY  http://localhost:{p}   (ACTUAL PORT {p}) | dropdown = switch prediction; "
          f"red=PW movers, yellow=GT object path", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
