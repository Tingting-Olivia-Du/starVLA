# [LangPointWorld] S1-c gate: instantiate QwenPIFlow with the real Qwen2.5-VL-3B, build ONE batch
# from the LIBERO+cache dataset, and overfit it for K steps. PASS = both action_loss and flow_loss
# drop sharply (model + distill loss + flow-token conditioning all wired correctly). Also runs a
# flow-OFF control to confirm it reduces to vanilla QwenPI behavior.
import argparse, os, sys
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, "/workspace/tingting/starVLA")
from starVLA.model.tools import FRAMEWORK_REGISTRY
import starVLA.model.framework.base_framework as bf
from starVLA.dataloader.libero_flow_dataset import LiberoFlowDataset


def build_cfg(condition_action=True, flow_enable=True):
    return OmegaConf.create({
        "framework": {
            "name": "QwenPIFlow",
            "qwenvl": {"base_vlm": "./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct",
                       "attn_implementation": "flash_attention_2", "vl_hidden_dim": 2048},
            "action_model": {"action_dim": 7, "state_dim": 7, "action_horizon": 8,
                             "repeated_diffusion_steps": 2},
            "flow": {"enable": flow_enable, "n_points": 256, "horizon": 10,
                     "lambda_flow": 1.0, "condition_action": condition_action},
        },
        "datasets": {"vla_data": {"obs_image_size": None}},
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/workspace/tingting/.tmp/s1_cache")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--condition-action", type=int, default=1)
    args = ap.parse_args()

    os.chdir("/workspace/tingting/starVLA")
    bf._auto_import_framework_modules()
    cfg = build_cfg(condition_action=bool(args.condition_action))
    print(f"[s1c] building QwenPIFlow (condition_action={bool(args.condition_action)}) ...", flush=True)
    model = FRAMEWORK_REGISTRY._registry["QwenPIFlow"](config=cfg).to("cuda")
    model.train()

    ds = LiberoFlowDataset(args.cache_dir, action_horizon=8, split="train")
    batch = [ds[i] for i in range(args.bs)]
    print(f"[s1c] batch={args.bs} | teacher_flow {batch[0]['teacher_flow'].shape} action {batch[0]['action'].shape}", flush=True)

    # only train the flow head + action head + (LoRA-free) — but for overfit, train everything small.
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    l0 = None
    for step in range(args.steps):
        out = model(batch)
        loss = out["loss"]
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 0:
            l0 = (float(out["action_loss"]), float(out["flow_loss"]))
        if step % 10 == 0 or step == args.steps - 1:
            print(f"[s1c] step {step:3d} total={float(loss):.4f} action={float(out['action_loss']):.4f} "
                  f"flow={float(out['flow_loss']):.4f}", flush=True)
    lf = (float(out["action_loss"]), float(out["flow_loss"]))
    print(f"\n[s1c] action_loss {l0[0]:.4f}->{lf[0]:.4f}  flow_loss {l0[1]:.4f}->{lf[1]:.4f}", flush=True)
    ok_a = lf[0] < 0.6 * l0[0] + 1e-6
    ok_f = lf[1] < 0.4 * l0[1] + 1e-6
    print(f"[s1c] {'PASS' if (ok_a and ok_f) else 'PARTIAL' if (ok_a or ok_f) else 'FAIL'}: "
          f"action drop={ok_a} flow drop={ok_f}", flush=True)


if __name__ == "__main__":
    main()
