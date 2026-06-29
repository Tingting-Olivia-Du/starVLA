# starVLA/model/modules/geomem/world_state_factory.py
# [D4RT-WorldState] Selects the geometric world-state + imaginer adapters on
# framework.world_state.backbone in {vggt_world, d4rt}. The two backbones share the
# adapter interface (encode/.flatten/.hidden_size, imagine_tokens/training_loss) so the
# GeoMemoryVLA call sites are unchanged.
# Part of the D4RT-WorldState comparison arm — see
# docs/superpowers/specs/2026-06-30-d4rt-worldstate-design.md
from __future__ import annotations

from starVLA.model.modules.geomem.world_state_adapter import WorldStateAdapter
from starVLA.model.modules.geomem.imagination_adapter import ImaginationAdapter
from starVLA.model.modules.geomem.d4rt_world_state_adapter import D4RTWorldStateAdapter
from starVLA.model.modules.geomem.d4rt_imagination_adapter import D4RTImaginationAdapter


def _world_state_cls(backbone: str):
    if backbone == "vggt_world":
        return WorldStateAdapter
    if backbone == "d4rt":
        return D4RTWorldStateAdapter
    raise ValueError(f"unknown world_state.backbone: {backbone!r}")


def build_world_state(fw):
    ws = fw.world_state
    backbone = ws.get("backbone", "vggt_world")
    if backbone == "vggt_world":
        return WorldStateAdapter(pretrained_vggt_repo=ws["model_name"])
    # d4rt
    return D4RTWorldStateAdapter(model_yaml=ws["d4rt_model_yaml"],
                                 ckpt_path=ws.get("d4rt_ckpt_path"))


def build_imaginer(fw, world_state):
    ws = fw.world_state
    imag = fw.imagination
    backbone = ws.get("backbone", "vggt_world")
    if backbone == "vggt_world":
        return ImaginationAdapter(
            pretrained_vggt_repo=ws["model_name"],
            chunk_size=int(imag["horizon"]),
            context_size=int(imag.get("context_size", 2)),
        )
    # d4rt
    return D4RTImaginationAdapter(
        world_state=world_state,
        subgoal_type=imag.get("subgoal_type", "latent"),
        horizon=int(imag["horizon"]),
    )
