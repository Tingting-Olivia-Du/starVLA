# [LangPointWorld] Interactive viser: GT grasped-object trajectory (red) vs PW predicted motion of
# near-object scene points (green), over the scene cloud + robot points. Makes the grasp-miss finding
# 3D-inspectable. Base frame throughout.
import argparse, time
import numpy as np
import viser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="/workspace/tingting/.tmp/sim_render_ep0.npz")
    ap.add_argument("--pred", default="/workspace/tingting/.tmp/sim_dualview_base_ep0.npz")
    ap.add_argument("--obj", default="alphabet_soup_1_main")
    ap.add_argument("--port", type=int, default=8084)
    args = ap.parse_args()

    sr = np.load(args.sim, allow_pickle=True)
    pr = np.load(args.pred, allow_pickle=True)
    names = list(sr["obj_names"])
    op = sr["obj_poses"]
    base_T = op[0][names.index("robot0_base")]; base_inv = np.linalg.inv(base_T)
    oidx = names.index(args.obj)

    sf = pr["__pred_scene_flows__"]                  # [T,Ns,3] base
    colors = pr["_scene_colors_u8"] if "_scene_colors_u8" in pr.files else pr["scene_colors"]
    robot = pr["robot_flows"]                        # [T,Nr,3] base
    obj_w = op[:, oidx, :3, 3]
    obj_b = (base_inv[:3, :3] @ obj_w.T).T + base_inv[:3, 3]   # [T,3] GT obj traj (base)
    near = np.linalg.norm(sf[0] - obj_b[0], axis=-1) < 0.08
    T = sf.shape[0]

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    gui_t = server.gui.add_slider("frame", 0, T - 1, 1, T - 1)
    gui_gt = server.gui.add_checkbox("GT object traj (red)", True)
    gui_pw = server.gui.add_checkbox("PW pred near object (green)", True)
    gui_rob = server.gui.add_checkbox("robot points (magenta)", True)

    def render():
        t = int(gui_t.value)
        server.scene.add_point_cloud("scene", sf[t].astype(np.float32),
                                     colors=colors[:sf.shape[1]].astype(np.uint8), point_size=0.004)
        server.scene.remove_by_name("gt_traj"); server.scene.remove_by_name("gt_now")
        if gui_gt.value:
            seg = np.stack([obj_b[:t] if t > 0 else obj_b[:1], obj_b[1:t+1] if t > 0 else obj_b[:1]], 1).astype(np.float32) if t > 0 else None
            if t > 0:
                segs = np.stack([obj_b[:t], obj_b[1:t+1]], 1).astype(np.float32)
                server.scene.add_line_segments("gt_traj", segs, colors=(255, 0, 0), line_width=5.0)
            server.scene.add_icosphere("gt_now", radius=0.02, color=(255, 0, 0), position=obj_b[t].astype(np.float32))
        server.scene.remove_by_name("pw_pred")
        if gui_pw.value and near.sum() and t > 0:
            segs = np.stack([sf[0][near], sf[t][near]], 1).astype(np.float32)
            server.scene.add_line_segments("pw_pred", segs, colors=(0, 230, 0), line_width=2.0)
        server.scene.remove_by_name("robot")
        if gui_rob.value:
            ti = min(t, robot.shape[0]-1)
            server.scene.add_point_cloud("robot", robot[ti].astype(np.float32),
                                         colors=(255, 0, 255), point_size=0.005)

    for g in (gui_t, gui_gt, gui_pw, gui_rob):
        g.on_update(lambda _: render())
    render()
    gt_disp = np.linalg.norm(obj_b[-1]-obj_b[0]); pw_disp = np.linalg.norm(sf[-1][near]-sf[0][near],axis=-1).mean() if near.sum() else 0
    print(f"[grasp-viser] READY http://localhost:{args.port} | {args.obj}: GT {gt_disp:.3f}m vs PW {pw_disp:.3f}m "
          f"| red=GT object traj, green=PW pred, magenta=robot", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
