# [LangPointWorld] S1-d: train the RGB-only QwenPIFlow student by distilling the frozen finetuned-PW
# teacher flow (cache) + behavior-cloning the LIBERO delta actions. Standalone DDP loop (torchrun,
# GPU 4,5). VLM frozen; flow head + action head trained. Logs action_loss / flow_loss / flow-EPE(m)
# on train+val; checkpoints (flow_head + action_model + config) to big disk. Re-runnable.
#
# Launch: torchrun --nproc_per_node=2 --master_port=29517 tools/pw_s1d_train.py --exp v1 ...
# See docs/superpowers/specs/2026-07-02-pointworld-s1-distill-architecture.md
import argparse, os, sys, time, json
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from omegaconf import OmegaConf

sys.path.insert(0, "/workspace/tingting/starVLA")
from starVLA.model.tools import FRAMEWORK_REGISTRY
import starVLA.model.framework.base_framework as bf
from starVLA.dataloader.libero_flow_dataset import LiberoFlowDataset, collate_examples


def cfg_for(args):
    return OmegaConf.create({
        "framework": {
            "name": "QwenPIFlow",
            "qwenvl": {"base_vlm": "./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct",
                       "attn_implementation": "flash_attention_2", "vl_hidden_dim": 2048},
            "action_model": {"action_dim": 7, "state_dim": 7, "action_horizon": 8,
                             "repeated_diffusion_steps": 2},
            "flow": {"enable": bool(args.flow_enable), "n_points": 256, "horizon": 10,
                     "lambda_flow": args.lambda_flow, "condition_action": bool(args.condition_action)},
        },
        "datasets": {"vla_data": {"obs_image_size": None}},
    })


@torch.no_grad()
def evaluate(model, loader, device, max_batches=20):
    model.eval()
    a_sum = f_sum = epe_sum = n = 0.0
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        out = model(batch)
        a_sum += float(out["action_loss"]); f_sum += float(out["flow_loss"]); n += 1
        # flow EPE in meters (final frame), rebuild teacher + pred via a light forward on the flow head
        core = model.module if hasattr(model, "module") else model
        imgs = [e["image"] for e in batch]; lang = [e["lang"] for e in batch]
        vl, mask = core._encode_vl_hidden_states(imgs, lang)
        fh = core._run_flow_head(vl[-1], mask)
        Ft = torch.tensor(np.array([e["teacher_flow"] for e in batch]), device=device, dtype=torch.float32)
        epe_sum += float((fh["flow"].float()[:, :, -1] - Ft[:, :, -1]).norm(dim=-1).mean())
    model.train()
    return a_sum / max(n, 1), f_sum / max(n, 1), epe_sum / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/workspace/tingting/.tmp/s1_cache")
    ap.add_argument("--out-dir", default="/workspace/tingting/.tmp/s1_train")
    ap.add_argument("--exp", default="v1")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda-flow", type=float, default=50.0)
    ap.add_argument("--condition-action", type=int, default=1)
    ap.add_argument("--flow-enable", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=2000)
    args = ap.parse_args()

    os.chdir("/workspace/tingting/starVLA")
    ddp = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if ddp and not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank); device = f"cuda:{local_rank}"
    is_main = rank == 0
    out_dir = os.path.join(args.out_dir, args.exp); os.makedirs(out_dir, exist_ok=True)

    bf._auto_import_framework_modules()
    model = FRAMEWORK_REGISTRY._registry["QwenPIFlow"](config=cfg_for(args)).to(device)
    # FREEZE VLM; train flow head + action head only
    for n_, p in model.named_parameters():
        p.requires_grad_(("flow_head" in n_) or ("action_model" in n_))
    if is_main:
        ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[s1d] exp={args.exp} trainable={ntrain/1e6:.1f}M lambda_flow={args.lambda_flow} "
              f"cond_action={bool(args.condition_action)} flow_enable={bool(args.flow_enable)}", flush=True)
    if ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    train_ds = LiberoFlowDataset(args.cache_dir, action_horizon=8, split="train")
    val_ds = LiberoFlowDataset(args.cache_dir, action_horizon=8, split="val")
    tsamp = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True) if ddp else None
    train_loader = DataLoader(train_ds, batch_size=args.bs, sampler=tsamp, shuffle=(tsamp is None),
                              collate_fn=collate_examples, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.bs, collate_fn=collate_examples, num_workers=2)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.05)

    step = 0; t0 = time.time(); best_epe = 1e9
    log_path = os.path.join(out_dir, "train_log.jsonl")
    while step < args.steps:
        if tsamp is not None:
            tsamp.set_epoch(step)
        for batch in train_loader:
            if step >= args.steps:
                break
            out = model(batch)
            loss = out["loss"]
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step()
            if is_main and step % 50 == 0:
                dt = (time.time() - t0) / max(step, 1)
                print(f"[s1d] step {step}/{args.steps} loss={float(loss):.3f} "
                      f"action={float(out['action_loss']):.3f} flow={float(out['flow_loss']):.5f} "
                      f"{dt:.2f}s/it", flush=True)
            if step % args.eval_every == 0 and step > 0:
                va, vf, vepe = evaluate(model, val_loader, device)
                if is_main:
                    rec = {"step": step, "val_action": va, "val_flow": vf, "val_flow_epe_m": vepe}
                    print(f"[s1d] EVAL step {step} val_action={va:.3f} val_flow={vf:.5f} val_flow_EPE={vepe:.4f}m", flush=True)
                    with open(log_path, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    if vepe < best_epe:
                        best_epe = vepe
                        core = model.module if hasattr(model, "module") else model
                        torch.save({"flow_head": core.flow_head.state_dict(),
                                    "action_model": core.action_model.state_dict(),
                                    "step": step, "val_flow_epe_m": vepe, "args": vars(args)},
                                   os.path.join(out_dir, "best.pt"))
                        print(f"[s1d] saved best.pt (val_flow_EPE {vepe:.4f}m)", flush=True)
            if is_main and step % args.save_every == 0 and step > 0:
                core = model.module if hasattr(model, "module") else model
                torch.save({"flow_head": core.flow_head.state_dict(),
                            "action_model": core.action_model.state_dict(), "step": step},
                           os.path.join(out_dir, f"step_{step}.pt"))
            step += 1
    if is_main:
        print(f"[s1d] DONE exp={args.exp} best_val_flow_EPE={best_epe:.4f}m", flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
