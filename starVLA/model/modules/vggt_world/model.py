# [Geo-MemoryVLA] Vendored verbatim from VGGT-World (Meta research license).
# Source: /workspace/tingting/vggt-world/vggt/world_model/model.py
# See docs/superpowers/specs/2026-06-29-geo-memoryvla-design.md
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from .backbone import FrozenVGGTBackbone
from .flow_model import TemporalFlowTransformer
from .scheduler import linear_ramp, sample_tau, sample_tau_mid
from .solver import euler_rollout
from .state import GeometryState, concat_states


def _to_long_tensor(value, device: torch.device) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = torch.as_tensor(value, device=device)
    if tensor.ndim == 0:
        tensor = tensor[None]
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.long()


class VGGTWorldModel(nn.Module):
    def __init__(
        self,
        vggt_model: Optional[nn.Module] = None,
        pretrained_vggt_repo: Optional[str] = None,
        state_dim: int = 1024,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        chunk_size: int = 2,
        context_size: int = 2,
        double_stream_depth: int = 8,
        single_stream_depth: int = 8,
        stage2_start: float = 0.5,
        rollout_steps_train: int = 8,
        rollout_steps_eval: int = 24,
        decode_during_train: bool = False,
    ):
        super().__init__()
        if vggt_model is None:
            if pretrained_vggt_repo is None:
                pretrained_vggt_repo = 'facebook/VGGT-1B'
            backbone = FrozenVGGTBackbone.from_pretrained(pretrained_vggt_repo)
        else:
            backbone = FrozenVGGTBackbone(vggt_model)
        self.backbone = backbone
        self.flow = TemporalFlowTransformer(
            dim=state_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            double_stream_depth=double_stream_depth,
            single_stream_depth=single_stream_depth,
        )
        self.chunk_size = chunk_size
        self.context_size = context_size
        self.stage2_start = stage2_start
        self.rollout_steps_train = rollout_steps_train
        self.rollout_steps_eval = rollout_steps_eval
        self.decode_during_train = decode_during_train

    def _state_positions(self, state: GeometryState) -> torch.Tensor:
        bsz, frames, tokens = state.frame_tokens.shape[:3]
        patch_h, patch_w = state.patch_grid
        base_pos = torch.zeros(bsz, frames, tokens, 3, device=state.device, dtype=torch.long)
        frame_ids = state.frame_ids
        if frame_ids is None:
            frame_ids = torch.arange(frames, device=state.device).view(1, frames).expand(bsz, -1)
        base_pos[..., 0] = frame_ids.unsqueeze(-1)
        patch_tokens = tokens - state.patch_start_idx
        if patch_tokens > 0:
            grid_y = torch.arange(patch_h, device=state.device)
            grid_x = torch.arange(patch_w, device=state.device)
            patch_pos = torch.cartesian_prod(grid_y, grid_x)[:patch_tokens]
            base_pos[:, :, state.patch_start_idx:, 1] = patch_pos[:, 0]
            base_pos[:, :, state.patch_start_idx:, 2] = patch_pos[:, 1]
        stacked = torch.cat([base_pos, base_pos], dim=2)
        return stacked.view(bsz, frames * stacked.shape[2], 3)

    def _flatten_state(self, state: GeometryState) -> torch.Tensor:
        return state.flatten_streams()

    def _noise_tokens(self, tokens: torch.Tensor, tau: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(tokens)
        while tau.ndim < tokens.ndim:
            tau = tau.unsqueeze(-1)
        noisy = (1.0 - tau) * tokens + tau * noise
        return noisy, noise

    def _state_from_pred_tokens(self, reference: GeometryState, pred_tokens: torch.Tensor) -> GeometryState:
        split = reference.num_tokens
        pred_tokens = pred_tokens.view(reference.batch_size, reference.num_frames, 2 * split, reference.dim)
        frame_tokens = pred_tokens[:, :, :split]
        global_tokens = pred_tokens[:, :, split:]
        return reference.with_updates(frame_tokens=frame_tokens, global_tokens=global_tokens)

    def _predict(self, target_state: GeometryState, condition_state: GeometryState, tau: torch.Tensor) -> torch.Tensor:
        target_pos = self._state_positions(target_state)
        condition_pos = self._state_positions(condition_state)
        target_noisy, _ = self._noise_tokens(self._flatten_state(target_state), tau)
        condition_flat = self._flatten_state(condition_state)
        return self.flow(
            target_noisy,
            tau=tau,
            condition=condition_flat,
            target_pos=target_pos,
            condition_pos=condition_pos,
        )

    def _partial_rollout(self, target_state: GeometryState, condition_state: GeometryState) -> GeometryState:
        target_pos = self._state_positions(target_state)
        condition_pos = self._state_positions(condition_state)
        noisy = torch.randn_like(self._flatten_state(target_state))
        tau_start = torch.ones(target_state.batch_size, device=target_state.device)
        tau_end = sample_tau_mid(target_state.batch_size, target_state.device)
        partial = euler_rollout(
            self.flow,
            noisy_state=noisy,
            condition=self._flatten_state(condition_state),
            target_pos=target_pos,
            condition_pos=condition_pos,
            tau_start=tau_start,
            tau_end=tau_end,
            steps=self.rollout_steps_train,
        )
        return self._state_from_pred_tokens(target_state, partial)

    def _attach_decoded_outputs(
        self,
        out: dict,
        decoded: dict,
        *,
        decoded_frame_start: int,
        batch_frame_start: int,
        decoded_frame_count: int,
        image_hw,
        decoded_frame_ids: torch.Tensor | None,
    ) -> None:
        out.update(decoded)
        out['decoded_frame_start'] = decoded_frame_start
        out['batch_frame_start'] = batch_frame_start
        out['decoded_frame_count'] = decoded_frame_count
        out['decoded_frame_ids'] = decoded_frame_ids
        if 'pose_enc' in decoded:
            out['decoded_pose_enc'] = decoded['pose_enc']
            extrinsics, intrinsics = pose_encoding_to_extri_intri(decoded['pose_enc'], image_hw)
            out['decoded_extrinsics'] = extrinsics
            out['decoded_intrinsics'] = intrinsics
        if 'depth' in decoded:
            out['decoded_depth'] = decoded['depth']
        if 'world_points' in decoded:
            out['decoded_world_points'] = decoded['world_points']

    def _stage1_forward(self, state: GeometryState) -> dict:
        condition = state.slice_frames(0, self.context_size)
        target = state.slice_frames(self.context_size, self.context_size + self.chunk_size)
        tau = sample_tau(state.batch_size, state.device)
        pred_tokens = self._predict(target, condition, tau)
        out = {
            'stage': 'stage1',
            'tau': tau,
            'condition_state_tokens': self._flatten_state(condition),
            'target_state_tokens': self._flatten_state(target),
            'pred_state_tokens': pred_tokens,
        }
        if self.decode_during_train:
            pred_state = self._state_from_pred_tokens(target, pred_tokens)
            decoded_state = concat_states([condition, pred_state])
            decoded = self.backbone.decode_geometry(decoded_state, images=None)
            self._attach_decoded_outputs(
                out,
                decoded,
                decoded_frame_start=condition.num_frames,
                batch_frame_start=self.context_size,
                decoded_frame_count=target.num_frames,
                image_hw=state.image_hw,
                decoded_frame_ids=decoded_state.frame_ids,
            )
        return out

    def _stage2_forward(self, state: GeometryState, where: float) -> dict:
        required_frames = self.context_size + self.chunk_size + 1
        if state.num_frames < required_frames:
            raise ValueError(
                f'Expected at least {required_frames} frames for Stage 2 flow forcing, got {state.num_frames}.'
            )

        history = state.slice_frames(0, self.context_size)
        first_target_start = self.context_size
        first_target = state.slice_frames(first_target_start, first_target_start + self.chunk_size)
        with torch.no_grad():
            partial = self._partial_rollout(first_target, history)
        mix_lambda = linear_ramp(0.0, 1.0, (where - self.stage2_start) / max(1e-6, 1.0 - self.stage2_start))

        mixed_frame = (1.0 - mix_lambda) * state.frame_tokens[:, first_target_start:first_target_start + 1] + mix_lambda * partial.frame_tokens[:, :1]
        mixed_global = (1.0 - mix_lambda) * state.global_tokens[:, first_target_start:first_target_start + 1] + mix_lambda * partial.global_tokens[:, :1]
        mixed_first_state = GeometryState(
            frame_tokens=mixed_frame,
            global_tokens=mixed_global,
            patch_start_idx=state.patch_start_idx,
            patch_grid=state.patch_grid,
            frame_ids=state.frame_ids[:, first_target_start:first_target_start + 1] if state.frame_ids is not None else None,
            image_hw=state.image_hw,
        )
        condition_parts = []
        if self.context_size > 1:
            condition_parts.append(state.slice_frames(1, self.context_size))
        condition_parts.append(mixed_first_state)
        mixed_condition = concat_states(condition_parts)

        second_target_start = self.context_size + 1
        second_target = state.slice_frames(second_target_start, second_target_start + self.chunk_size)
        tau = sample_tau(state.batch_size, state.device)
        pred_tokens = self._predict(second_target, mixed_condition, tau)
        out = {
            'stage': 'stage2',
            'tau': tau,
            'mix_lambda': torch.full((state.batch_size,), mix_lambda, device=state.device, dtype=state.dtype),
            'condition_state_tokens': self._flatten_state(mixed_condition),
            'target_state_tokens': self._flatten_state(second_target),
            'pred_state_tokens': pred_tokens,
            'aux_partial_state_tokens': self._flatten_state(partial),
        }
        if self.decode_during_train:
            pred_state = self._state_from_pred_tokens(second_target, pred_tokens)
            decoded_state = concat_states([mixed_condition, pred_state])
            decoded = self.backbone.decode_geometry(decoded_state, images=None)
            self._attach_decoded_outputs(
                out,
                decoded,
                decoded_frame_start=mixed_condition.num_frames,
                batch_frame_start=second_target_start,
                decoded_frame_count=second_target.num_frames,
                image_hw=state.image_hw,
                decoded_frame_ids=decoded_state.frame_ids,
            )
        return out

    def _next_frame_ids(self, current: GeometryState, target_frames: int) -> torch.Tensor | None:
        if current.frame_ids is None:
            return None
        last_ids = current.frame_ids[:, -1:]
        if current.num_frames > 1:
            step = current.frame_ids[:, -1:] - current.frame_ids[:, -2:-1]
            step = torch.where(step == 0, torch.ones_like(step), step)
        else:
            step = torch.ones_like(last_ids)
        increments = torch.arange(1, target_frames + 1, device=current.device, dtype=torch.long).view(1, target_frames)
        return last_ids + step * increments

    def _forecast(self, state: GeometryState, images: torch.Tensor | None, forecast_frames: Optional[int] = None) -> dict:
        current = state.slice_frames(0, self.context_size)
        total_future = forecast_frames if forecast_frames is not None else max(0, state.num_frames - self.context_size)
        generated_frames = 0
        predictions = []
        while generated_frames < total_future:
            target_frames = min(self.chunk_size, total_future - generated_frames)
            dummy_target = GeometryState(
                frame_tokens=torch.randn(current.batch_size, target_frames, current.num_tokens, current.dim, device=current.device, dtype=current.dtype),
                global_tokens=torch.randn(current.batch_size, target_frames, current.num_tokens, current.dim, device=current.device, dtype=current.dtype),
                patch_start_idx=current.patch_start_idx,
                patch_grid=current.patch_grid,
                frame_ids=self._next_frame_ids(current, target_frames),
                image_hw=current.image_hw,
            )
            pred_tokens = euler_rollout(
                self.flow,
                noisy_state=self._flatten_state(dummy_target),
                condition=self._flatten_state(current),
                target_pos=self._state_positions(dummy_target),
                condition_pos=self._state_positions(current),
                tau_start=torch.ones(current.batch_size, device=current.device),
                tau_end=torch.zeros(current.batch_size, device=current.device),
                steps=self.rollout_steps_eval,
            )
            pred_state = self._state_from_pred_tokens(dummy_target, pred_tokens)
            predictions.append(pred_state)
            combined = concat_states([current, pred_state])
            current = combined.slice_frames(max(0, combined.num_frames - self.context_size), combined.num_frames)
            generated_frames += target_frames
        forecast_state = concat_states([state.slice_frames(0, self.context_size)] + predictions) if predictions else state.slice_frames(0, self.context_size)
        decoded = self.backbone.decode_geometry(forecast_state, images)
        out = {
            'forecast_state_tokens': self._flatten_state(forecast_state),
            'target_state_tokens': self._flatten_state(state.slice_frames(self.context_size, self.context_size + total_future)),
            'pred_state_tokens': self._flatten_state(forecast_state.slice_frames(self.context_size, forecast_state.num_frames)),
        }
        self._attach_decoded_outputs(
            out,
            decoded,
            decoded_frame_start=self.context_size,
            batch_frame_start=self.context_size,
            decoded_frame_count=total_future,
            image_hw=state.image_hw if state.image_hw is not None else images.shape[-2:],
            decoded_frame_ids=forecast_state.frame_ids,
        )
        return out

    def forward(
        self,
        images: torch.Tensor,
        batch=None,
        phase: Optional[str] = None,
        where: float = 0.0,
        forecast_frames: Optional[int] = None,
        **kwargs,
    ):
        state = self.backbone.encode_states(images)
        if batch is not None and 'ids' in batch:
            frame_ids = _to_long_tensor(batch['ids'], state.device)
            if frame_ids is not None and frame_ids.shape[1] == state.num_frames:
                state = state.with_updates(frame_ids=frame_ids)
        if not self.training:
            if forecast_frames is None:
                if batch is not None and 'forecast_frames' in batch:
                    forecast_frames = int(batch['forecast_frames'][0]) if hasattr(batch['forecast_frames'], '__len__') else int(batch['forecast_frames'])
                else:
                    forecast_frames = max(0, images.shape[1] - self.context_size)
            return self._forecast(state, images, forecast_frames=forecast_frames)
        if state.num_frames < self.context_size + self.chunk_size:
            raise ValueError(
                f'Expected at least {self.context_size + self.chunk_size} frames for Stage 1 training, got {state.num_frames}.'
            )
        if state.num_frames >= self.context_size + self.chunk_size + 1 and where >= self.stage2_start:
            return self._stage2_forward(state, where)
        return self._stage1_forward(state)
