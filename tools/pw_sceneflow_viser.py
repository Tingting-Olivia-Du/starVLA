# [LangPointWorld] Show the EXACT scene_flows that go INTO PointWorld at inference, in TRUE color,
# no highlight. This is build_from_sim's output (the tensor PW's forward consumes): single agentview
# (matches the S0 packer / S1 teacher-cache convention), world frame, t=0 RGB-D cloud tiled over T.
# Time slider animates the frames (they're static-init copies of t=0 — PW is what predicts motion;
# this shows the INPUT, not any prediction). Use --dual to instead show the merged 2-view cloud.
import argparse, os, sys, time, socket
import numpy as np
import h5py

sys.path.insert(0, "/workspace/tingting/envs/pw_extra_site")
sys.path.insert(0, "/workspace/tingting/starVLA")

DATA = "/workspace/tingting/LIBERO/libero/datasets/libero_object"
T_MODEL = 11


def free_port(p):
    while True:
        s = socket.socket(); busy = s.connect_ex(("127.0.0.1", p)) == 0; s.close()
        if not busy:
            return p
        p += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True)
    ap.add_argument("--single", action="store_true", help="single agentview only (default is DUAL-view merged: agentview+wrist)")
    ap.add_argument("--frame", default="world", choices=["world", "base"])
    ap.add_argument("--max-points", type=int, default=16384,
                    help="scene point budget (density). S0 train/packer + build_from_sim default=8192 "
                         "(matches official PW ~10000); raise for denser viz. 32768=~all pixels (2x16384).")
    ap.add_argument("--port", type=int, default=8090)
    args = ap.parse_args()

    from starVLA.model.modules.langpw.libero_to_datadict import LiberoDataDictBuilder
    base = os.path.basename(args.sim)[:-4]
    demo = base.split("_demo_")[1]; task = base.rsplit("_demo_", 1)[0]
    H = f"{DATA}/pick_up_the_{task}_and_place_it_in_the_basket_demo.hdf5"
    sr = np.load(args.sim, allow_pickle=True)
    with h5py.File(H, "r") as f:
        o = f["data"][f"demo_{demo}"]["obs"]
        joints = np.asarray(o["joint_states"][:], np.float64)
        grip = np.asarray(o["gripper_states"][:], np.float64)

    cams = ("agentview",) if args.single else ("agentview", "robot0_eye_in_hand")  # DUAL by default
    builder = LiberoDataDictBuilder(domain="droid", device="cuda", max_scene_points=args.max_points)
    dd = builder.build_from_sim(sr, joints, grip, horizon=T_MODEL, cams=cams, frame=args.frame)

    sf = dd["scene_flows"]                                  # [T, Ns, 3]  <-- THE PW INPUT
    cols = dd["_scene_colors_u8"][:sf.shape[1]]             # true RGB per point
    rf = dd["robot_flows"]                                 # [T, Nr, 3]  <-- THE PW ROBOT/ACTION INPUT
    T, Ns, _ = sf.shape
    print(f"[scene] task={task} demo_{demo} {'SINGLE agentview' if args.single else 'DUAL-view (agentview+wrist)'} frame={args.frame}", flush=True)
    print(f"[scene] scene_flows {sf.shape}  (PW scene input)", flush=True)
    print(f"[scene] robot_flows {rf.shape}  (PW action/proprio input = URDF-FK point cloud; moves over t)", flush=True)

    import viser
    p = free_port(args.port)
    server = viser.ViserServer(port=p)
    server.scene.set_up_direction("+z")
    gui_t = server.gui.add_slider("frame", 0, T - 1, 1, 0)
    gui_scene = server.gui.add_checkbox("scene_flows (true color)", True)
    gui_robot = server.gui.add_checkbox("robot_flows (FK, magenta)", True)

    def render():
        t = int(gui_t.value)
        if gui_scene.value:
            server.scene.add_point_cloud("scene_flow", sf[t].astype(np.float32),
                                         colors=cols.astype(np.uint8), point_size=0.004)
        else:
            server.scene.remove_by_name("scene_flow")
        if gui_robot.value:
            ti = min(t, rf.shape[0] - 1)
            server.scene.add_point_cloud("robot_flow", rf[ti].astype(np.float32),
                                         colors=(255, 0, 255), point_size=0.006)
        else:
            server.scene.remove_by_name("robot_flow")
    for g in (gui_t, gui_scene, gui_robot):
        g.on_update(lambda _: render())
    render()
    rmot = np.linalg.norm(rf[-1].mean(0) - rf[0].mean(0))
    print(f"[scene] viser READY  http://localhost:{p}   (ACTUAL PORT {p}) | scene=true color, "
          f"robot=magenta (moves {rmot:.3f}m over episode); drag slider to animate the FK arm", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
