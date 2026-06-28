# starVLA/model/modules/geomem/imagination_adapter.py
# [Geo-MemoryVLA] Adapter over the vendored VGGT-World VGGTWorldModel. Produces the
# imagination training loss and the imagined future GeometryState (3D visual subgoal).
# All flow-matching / z-prediction / flow-forcing logic lives in modules/vggt_world.
# Part of the Geo-MemoryVLA architecture — see
# docs/superpowers/specs/2026-06-29-geo-memoryvla-design.md
from __future__ import annotations

import torch
import torch.nn as nn

from starVLA.model.modules.vggt_world.losses import WorldModelLoss
from starVLA.model.modules.vggt_world.model import VGGTWorldModel


class ImaginationAdapter(nn.Module):
    def __init__(
        self,
        pretrained_vggt_repo: str = "facebook/VGGT-1B",
        chunk_size: int = 2,
        context_size: int = 2,
        latent_weight: float = 1.0,
        decode_weights: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        super().__init__()
        self.world_model = VGGTWorldModel(
            pretrained_vggt_repo=pretrained_vggt_repo,
            chunk_size=chunk_size,
            context_size=context_size,
        )
        d_depth, d_point, d_cam = decode_weights
        self.criterion = WorldModelLoss(
            latent_weight=latent_weight,
            decode_depth_weight=d_depth,
            decode_point_weight=d_point,
            decode_camera_weight=d_cam,
        )

    def training_loss(self, images: torch.Tensor, where: float = 0.0) -> torch.Tensor:
        # [Geo-MemoryVLA] NOTE: `images` MUST be a multi-frame window of shape
        # [B, F, 3, H, W] with F >= context_size + chunk_size for Stage 1 training
        # (and F >= context_size + chunk_size + 1 for Stage 2 flow-forcing when
        # where >= stage2_start). The vendored VGGTWorldModel.forward() will raise
        # ValueError if the frame count is insufficient. A single-frame input
        # (F=1) will fail. Building the multi-frame window is Task 6's
        # responsibility via _build_image_window — this adapter only passes
        # `images` through to the world model unchanged.
        self.world_model.train()
        preds = self.world_model(images, where=where)
        return self.criterion(preds, batch={})["objective"]

    @torch.no_grad()
    def imagine_tokens(self, images: torch.Tensor, forecast_frames: int) -> torch.Tensor:
        # [Geo-MemoryVLA] NOTE: `images` MUST be a multi-frame window of shape
        # [B, F, 3, H, W]. The vendored VGGTWorldModel.forward() in eval mode
        # calls _forecast(), which uses the first context_size frames as the
        # conditioning window. A single-frame input (F=1) will produce a
        # degenerate forecast (context=target=current frame) with no meaningful
        # 3D subgoal signal. Task 6's _build_image_window is responsible for
        # assembling the correct multi-frame window before calling this method.
        self.world_model.eval()
        out = self.world_model(images, forecast_frames=forecast_frames)
        # pred_state_tokens: flattened future tokens [B, L_img, 1024].
        return out["pred_state_tokens"]
