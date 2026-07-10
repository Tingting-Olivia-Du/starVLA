# [LangPointWorld] Export a trained QwenPIFlow run (best.pt = flow_head + action_model, VLM frozen)
# into a full from_pretrained-loadable checkpoint dir for the eval server: rebuilds the model (base
# VLM + trained heads), writes the FULL state_dict + a config.yaml next to it. So closed-loop LIBERO
# eval (deployment/model_server, baseframework.from_pretrained) can load the RGB-only student.
import argparse, os, sys
import torch
from omegaconf import OmegaConf

sys.path.insert(0, "/workspace/tingting/starVLA")
from starVLA.model.tools import FRAMEWORK_REGISTRY
import starVLA.model.framework.base_framework as bf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="e.g. /workspace/tingting/.tmp/s1_train/v1_flow_cond")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--condition-action", type=int, default=1)
    ap.add_argument("--flow-enable", type=int, default=1)
    args = ap.parse_args()

    os.chdir("/workspace/tingting/starVLA")
    bf._auto_import_framework_modules()
    cfg = OmegaConf.create({
        "framework": {"name": "QwenPIFlow",
                      "qwenvl": {"base_vlm": "./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct",
                                 "attn_implementation": "flash_attention_2", "vl_hidden_dim": 2048},
                      "action_model": {"action_dim": 7, "state_dim": 7, "action_horizon": 8,
                                       "repeated_diffusion_steps": 2},
                      "flow": {"enable": bool(args.flow_enable), "n_points": 256, "horizon": 10,
                               "lambda_flow": 50.0, "condition_action": bool(args.condition_action)}},
        "datasets": {"vla_data": {"data_mix": "libero_object", "obs_image_size": None,
                                  "data_root_dir": "playground/Datasets/LEROBOT_LIBERO_DATA"}},
        "trainer": {"pretrained_checkpoint": None},
    })
    model = FRAMEWORK_REGISTRY._registry["QwenPIFlow"](config=cfg)
    trained = torch.load(os.path.join(args.run_dir, args.ckpt), map_location="cpu")
    model.flow_head.load_state_dict(trained["flow_head"])
    model.action_model.load_state_dict(trained["action_model"])

    # Layout the eval server (baseframework.from_pretrained -> read_mode_config) expects:
    #   <run_dir>/checkpoints/model.pt   +   <run_dir>/config.yaml   +   <run_dir>/dataset_statistics.json
    import json
    export_root = os.path.join(args.run_dir, "export"); ckpt_dir = os.path.join(export_root, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
    OmegaConf.save(cfg, os.path.join(export_root, "config.yaml"))
    # IDENTITY action stats: the student was trained on RAW LIBERO [-1,1] delta actions (no norm),
    # so eval must NOT re-normalize. Identity mean0/std1/min-1/max1 => norm/denorm are no-ops.
    d = 7; ident = {"franka": {"action": {"mean": [0.0]*d, "std": [1.0]*d, "min": [-1.0]*d, "max": [1.0]*d,
                                          "q01": [-1.0]*d, "q99": [1.0]*d},
                               "proprio": {"mean": [0.0]*d, "std": [1.0]*d, "min": [-1.0]*d, "max": [1.0]*d}}}
    with open(os.path.join(export_root, "dataset_statistics.json"), "w") as f:
        json.dump(ident, f, indent=2)
    print(f"[export] wrote {ckpt_dir}/model.pt + config.yaml + dataset_statistics.json (step "
          f"{trained.get('step','?')}, val_flow_EPE {trained.get('val_flow_epe_m','?')})", flush=True)
    print(f"[export] closed-loop: baseframework.from_pretrained('{ckpt_dir}/model.pt'). Needs kami "
          f"LIBERO eval box. CAVEAT: action space = raw [-1,1] delta w/ IDENTITY stats — verify the "
          f"eval env's action convention matches. See [[starvla-libero-eval-setup]].", flush=True)


if __name__ == "__main__":
    main()
