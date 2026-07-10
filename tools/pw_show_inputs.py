# [LangPointWorld] Show EXACTLY what goes INTO PointWorld for a LIBERO episode, built from GT depth
# (sim-render) + RGB — NO VGGT. Dumps + visualizes the PW input tensors:
#   INPUT RGB-D:   agentview RGB (point colors) + GT metric depth (colormap)
#   scene_flows[T,Ns,3]: t=0 scene point cloud (RGB-D unprojected, world frame) tiled over T. This is
#                        the "static-init" the model OVERWRITES with its imagined future motion.
#   robot_flows[T,Nr,3]: robot point cloud over time from URDF forward-kinematics on the joint traj —
#                        the ACTION PW conditions on.
# Produces: (a) a PNG panel [RGB | GT depth | scene t=0 reprojected | robot t0->tT reprojected], and
# (b) a viser 3D scene (scene cloud in true colors + robot cloud animated) so you can rotate/inspect.
#
# Run in starVLA env. Example:
#   CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/workspace/tingting/envs/pw_extra_site:/workspace/tingting/starVLA \
#   python tools/pw_show_inputs.py --sim /workspace/tingting/.tmp/s0_sim/alphabet_soup_demo_0.npz --viser 0
import argparse, os, sys
import numpy as np
import h5py
import cv2

sys.path.insert(0, "/workspace/tingting/envs/pw_extra_site")
sys.path.insert(0, "/workspace/tingting/starVLA")

DATA = "/workspace/tingting/LIBERO/libero/datasets/libero_object"
T_MODEL = 11


def colormap_depth(d):
    m = np.isfinite(d) & (d > 1e-6) & (d < 5)
    vmin = np.percentile(d[m], 2) if m.any() else 0
    vmax = np.percentile(d[m], 98) if m.any() else 1
    n = np.clip((d - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    col = cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    col[~m] = 0
    return col, float(vmin), float(vmax)


def project(P, K, E_w2c):
    o = np.ones((P.shape[0], 1)); c = (E_w2c @ np.concatenate([P, o], -1).T).T[:, :3]
    z = c[:, 2]; uv = (K @ c.T).T; uv = uv[:, :2] / np.clip(uv[:, 2:3], 1e-6, None)
    return uv, z


def splat(canvas, uv, z, colors, r=2):
    Hc, Wc = canvas.shape[:2]
    for i in np.argsort(-z):
        x, y = int(uv[i, 0]), int(uv[i, 1])
        if 0 <= x < Wc and 0 <= y < Hc and z[i] > 1e-4:
            c = colors[i]
            cv2.circle(canvas, (x, y), r, (int(c[2]), int(c[1]), int(c[0])), -1)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True, help="sim-render npz (GT depth/K/E)")
    ap.add_argument("--out-dir", default="/workspace/tingting/.tmp/pw_inputs")
    ap.add_argument("--frame", default="world", choices=["world", "base"])
    ap.add_argument("--cams", default="agentview,robot0_eye_in_hand",
                    help="comma list of sim cameras; DEFAULT dual-view (build_from_sim default). "
                         "S0 finetune-data was SINGLE agentview — pass --cams agentview to match that.")
    ap.add_argument("--viser", type=int, default=0, help="1 = also launch viser")
    ap.add_argument("--port", type=int, default=8084)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cams = tuple(args.cams.split(","))

    from starVLA.model.modules.langpw.libero_to_datadict import LiberoDataDictBuilder

    base = os.path.basename(args.sim)[:-4]
    demo = base.split("_demo_")[1]; task = base.rsplit("_demo_", 1)[0]
    H = f"{DATA}/pick_up_the_{task}_and_place_it_in_the_basket_demo.hdf5"
    sr = np.load(args.sim, allow_pickle=True)
    with h5py.File(H, "r") as f:
        o = f["data"][f"demo_{demo}"]["obs"]
        joints = np.asarray(o["joint_states"][:], np.float64)
        grip = np.asarray(o["gripper_states"][:], np.float64)

    builder = LiberoDataDictBuilder(domain="droid", device="cuda")
    dd = builder.build_from_sim(sr, joints, grip, horizon=T_MODEL, cams=cams, frame=args.frame)

    # ---- report the actual PW input tensors ----
    sf = dd["scene_flows"]; rf = dd["robot_flows"]
    ncam = len(cams)
    print(f"[pw-input] task={task} demo_{demo} frame={args.frame} cams={cams} ({ncam}-view)", flush=True)
    ncam_dd = sum(1 for k in dd if k.endswith("_initial_rgb"))
    print(f"[pw-input] PW cam payloads in data_dict: {ncam_dd} (cam0..cam{ncam_dd-1}) — multi-view scene encoder", flush=True)
    print(f"[pw-input] scene_flows   {sf.shape}  (T, Ns, 3)  t=0 RGB-D cloud, tiled; model predicts motion", flush=True)
    print(f"[pw-input] robot_flows   {rf.shape}  (T, Nr, 3)  URDF-FK action cloud over the joint traj", flush=True)
    print(f"[pw-input] scene_features{dd['scene_features'].shape}  robot_features{dd['robot_features'].shape}", flush=True)
    print(f"[pw-input] cam0 K/E/depth present: {all(k in dd for k in ('cam0_intrinsic','cam0_extrinsic','cam0_initial_depth'))}", flush=True)
    print(f"[pw-input] robot motion over episode: {np.linalg.norm(rf[-1].mean(0)-rf[0].mean(0)):.3f} m (EE region moves)", flush=True)

    # ---- ROW 1: per-view RGB + GT depth (raw, as stored) with the UP/DOWN FLIP shown explicitly ----
    up = 3; H0, W0 = 128, 128; Hc, Wc = 180 * up, 320 * up
    tile_h = H0 * 2  # small per-view tiles at 2x
    row1_tiles = []
    for c in cams:
        rgb_raw = sr[f"{c}_rgb"][0]                       # as stored (opengl-flipped, i.e. up/down)
        dep_raw = sr[f"{c}_depth"][0].astype(np.float32)
        rgb_flip = rgb_raw[::-1]                          # after [::-1] flip used before unprojection
        dep_flip = dep_raw[::-1]
        dcol_raw, _, _ = colormap_depth(dep_raw)
        dcol_flip, vmn, vmx = colormap_depth(dep_flip)
        def _t(img, txt):
            u = cv2.resize(img, (W0 * 2, H0 * 2), interpolation=cv2.INTER_NEAREST)
            cv2.putText(u, txt, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            return u
        # show raw (as-stored, flipped) vs corrected (after [::-1], right-side-up used for unproject)
        blk = np.hstack([_t(rgb_raw[:, :, ::-1], f"{c[:10]} RGB raw"),
                         _t(rgb_flip[:, :, ::-1], f"{c[:10]} RGB [::-1] (used)"),
                         _t(dcol_flip, f"{c[:10]} depth [{vmn:.2f},{vmx:.2f}]m")])
        row1_tiles.append(blk)
    row1 = np.vstack(row1_tiles)

    # ---- ROW 2: merged multi-view scene cloud (per-view colored) + robot flow, reprojected into cam0 ----
    K = dd["cam0_intrinsic"].astype(np.float64); E = dd["cam0_extrinsic"].astype(np.float64)  # scene->cam0
    scene0 = sf[0]; scols = dd["_scene_colors_u8"][:scene0.shape[0]]
    Ku = K.copy(); Ku[:2, :] *= up

    sc = np.full((Hc, Wc, 3), 30, np.uint8)
    uv, z = project(scene0, Ku, E); sc = splat(sc, uv, z, scols, r=2)
    cv2.putText(sc, f"scene_flows[0] MERGED {ncam}-view RGB-D cloud (Ns={scene0.shape[0]})",
                (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    rb = np.full((Hc, Wc, 3), 30, np.uint8)
    for t, col in [(0, (255, 0, 255)), (T_MODEL - 1, (0, 255, 255))]:
        uv, z = project(rf[t], Ku, E)
        rb = splat(rb, uv, z, np.tile(np.array([col]), (rf.shape[1], 1)), r=2)
    cv2.putText(rb, "robot_flows t0(magenta)->tT(yellow)", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    row2 = np.hstack([sc, rb])

    # pad row1 to row2 width
    if row1.shape[1] < row2.shape[1]:
        row1 = cv2.copyMakeBorder(row1, 0, 0, 0, row2.shape[1] - row1.shape[1], cv2.BORDER_CONSTANT, value=0)
    elif row1.shape[1] > row2.shape[1]:
        row2 = cv2.copyMakeBorder(row2, 0, 0, 0, row1.shape[1] - row2.shape[1], cv2.BORDER_CONSTANT, value=0)
    fig = np.vstack([row1, row2])
    outp = os.path.join(args.out_dir, f"{task}_demo_{demo}_pw_inputs.png")
    cv2.imwrite(outp, fig)
    np.savez(os.path.join(args.out_dir, f"{task}_demo_{demo}_pw_inputs.npz"),
             scene_flows=sf, robot_flows=rf, scene_colors=dd["_scene_colors_u8"],
             cam0_K=K, cam0_E=E, cams=np.array(cams))
    print(f"[pw-input] wrote {outp}", flush=True)

    if args.viser:
        import viser, time
        server = viser.ViserServer(port=args.port); server.scene.set_up_direction("+z")
        gui_t = server.gui.add_slider("robot frame", 0, T_MODEL - 1, 1, 0)
        server.scene.add_point_cloud("scene", scene0.astype(np.float32),
                                     colors=scols.astype(np.uint8), point_size=0.004)

        def render():
            t = int(gui_t.value)
            server.scene.add_point_cloud("robot", rf[t].astype(np.float32),
                                         colors=(255, 0, 255), point_size=0.006)
        gui_t.on_update(lambda _: render()); render()
        print(f"[pw-input] viser READY http://localhost:{args.port} (scene=true color, robot=magenta, slider=time)", flush=True)
        while True:
            time.sleep(3600)


if __name__ == "__main__":
    main()
