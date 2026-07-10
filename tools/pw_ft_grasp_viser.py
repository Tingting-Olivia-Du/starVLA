# [LangPointWorld] Interactive viser for a FINETUNED-model prediction npz (world frame). Scene cloud
# in true colors, predicted flow of the top-K movers (green lines t0->t), robot cloud (magenta), GT
# object trajectory (yellow). Frame slider + episode dropdown. Confirms the finetuned model carries
# the grasped object in 3D.
import argparse, glob, os, time
import numpy as np
import viser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="/workspace/tingting/.tmp/s0_ftviz/pred_*.npz")
    ap.add_argument("--port", type=int, default=8084)
    ap.add_argument("--topk", type=int, default=300)
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    assert files, f"no npz matched {args.glob}"
    tags = [os.path.basename(f)[5:-4] for f in files]

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    gui_ep = server.gui.add_dropdown("episode", tags, tags[0])
    gui_t = server.gui.add_slider("frame", 0, 10, 1, 10)
    gui_pw = server.gui.add_checkbox("pred flow movers (green)", True)
    gui_gt = server.gui.add_checkbox("GT object traj (yellow)", True)
    gui_rob = server.gui.add_checkbox("robot (magenta)", True)

    state = {}

    def load(tag):
        z = np.load(files[tags.index(tag)], allow_pickle=True)
        pred = z["__pred_scene_flows__"].astype(np.float32)
        col = z["_scene_colors_u8"].astype(np.uint8)
        robot = z["robot_flows"].astype(np.float32)
        obj = z["obj_traj_world"].astype(np.float32)
        disp = np.linalg.norm(pred[-1] - pred[0], axis=-1)
        kp = np.argsort(disp)[-args.topk:]
        state.update(dict(pred=pred, col=col, robot=robot, obj=obj, kp=kp, gt=float(z["gt_disp"]), tag=tag))
        gui_t.max = pred.shape[0] - 1

    def render():
        s = state; t = int(gui_t.value)
        pred, col, kp = s["pred"], s["col"], s["kp"]
        server.scene.add_point_cloud("scene", pred[t], colors=col[:pred.shape[1]], point_size=0.004)
        for nm in ("pw", "gt_traj", "gt_now", "robot"):
            server.scene.remove_by_name(nm)
        if gui_pw.value and t > 0:
            segs = np.stack([pred[0][kp], pred[t][kp]], 1)
            server.scene.add_line_segments("pw", segs, colors=(0, 230, 0), line_width=2.0)
        if gui_gt.value:
            if t > 0:
                segs = np.stack([s["obj"][:t], s["obj"][1:t + 1]], 1)
                server.scene.add_line_segments("gt_traj", segs, colors=(255, 255, 0), line_width=5.0)
            server.scene.add_icosphere("gt_now", radius=0.02, color=(255, 200, 0), position=s["obj"][t])
        if gui_rob.value:
            ti = min(t, s["robot"].shape[0] - 1)
            server.scene.add_point_cloud("robot", s["robot"][ti], colors=(255, 0, 255), point_size=0.005)

    def on_ep(_):
        load(gui_ep.value); render()

    load(tags[0])
    for g in (gui_t, gui_pw, gui_gt, gui_rob):
        g.on_update(lambda _: render())
    gui_ep.on_update(on_ep)
    render()
    print(f"[ft-viser] READY http://localhost:{args.port} | {len(files)} episodes | "
          f"green=pred flow, yellow=GT obj, magenta=robot", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
