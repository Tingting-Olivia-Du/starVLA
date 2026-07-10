# [LangPointWorld] S1-c: QwenPIFlow — QwenPI + point-flow distillation head.
# Subclasses Qwen_PI. Adds a PointFlowHead that predicts N-point future flow from the VLM's last
# hidden state; distills it against the frozen finetuned-PW teacher cache (pw_distill_loss); and
# (optionally) feeds the flow tokens into the action head as extra cross-attention memory. The
# flow.condition_action flag is the S1 ablation switch (is flow decorative?). RGB-only at test time.
#
# See docs/superpowers/specs/2026-07-02-pointworld-s1-distill-architecture.md
from typing import List, Optional, Tuple

import numpy as np
import torch

from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI, QwenPIDefaultConfig
from starVLA.model.modules.langpw.point_flow_head import PointFlowHead
from starVLA.model.modules.langpw.pw_distill_loss import pw_distill_loss
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


@FRAMEWORK_REGISTRY.register("QwenPIFlow")
class Qwen_PIFlow(Qwen_PI):
    """QwenPI + distilled point-flow head. Config additions (framework.flow.*):
        enable            (bool)  master switch (default True)
        n_points          (int)   N teacher/student points (default 256)
        horizon           (int)   Hf flow horizon (default 10 = teacher T-1)
        lambda_flow       (float) weight of the distill loss in the total loss (default 1.0)
        condition_action  (bool)  feed flow_tokens into the action head (default True; ABLATION)
    """

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        fcfg = self.config.framework.get("flow", {}) if self.config and hasattr(self.config, "framework") else {}
        self.flow_enable = bool(fcfg.get("enable", True))
        self.flow_n_points = int(fcfg.get("n_points", 256))
        self.flow_horizon = int(fcfg.get("horizon", 10))
        self.lambda_flow = float(fcfg.get("lambda_flow", 1.0))
        self.flow_condition_action = bool(fcfg.get("condition_action", True))

        D = int(self.config.framework.qwenvl.vl_hidden_dim)  # LLM hidden size
        # hidden == D so flow_tokens concat directly into the action head's cross-attn memory.
        self.flow_head = PointFlowHead(
            cond_dim=D, n_points=self.flow_n_points, flow_horizon=self.flow_horizon, hidden=D
        )
        logger.info(f"[QwenPIFlow] flow head N={self.flow_n_points} Hf={self.flow_horizon} "
                    f"lambda={self.lambda_flow} condition_action={self.flow_condition_action}")

    def _run_flow_head(self, base_hidden, backbone_attention_mask):
        """base_hidden [B,L,D] -> {flow[B,N,Hf,3], flow_tokens[B,N,D]}."""
        enc_mask = None
        if backbone_attention_mask is not None:
            enc_mask = backbone_attention_mask.to(dtype=torch.bool)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fh = self.flow_head(base_hidden, encoder_attention_mask=enc_mask)
        return fh

    def _augment_vl_with_flow(self, vl_embs_list, backbone_attention_mask, flow_tokens):
        """Append flow_tokens [B,N,D] to every layer's hidden states + extend the attention mask."""
        ft = flow_tokens.to(vl_embs_list[0].dtype)
        aug = [torch.cat([h, ft], dim=1) for h in vl_embs_list]
        if backbone_attention_mask is not None:
            B, N = ft.shape[0], ft.shape[1]
            ones = torch.ones(B, N, dtype=backbone_attention_mask.dtype, device=backbone_attention_mask.device)
            backbone_attention_mask = torch.cat([backbone_attention_mask, ones], dim=1)
        return aug, backbone_attention_mask

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        vl_embs_list, backbone_attention_mask = self._encode_vl_hidden_states(batch_images, instructions)
        base_hidden = vl_embs_list[-1]
        device, dtype = base_hidden.device, base_hidden.dtype

        # ---- flow head + distillation loss ----
        flow_loss = torch.zeros((), device=device, dtype=torch.float32)
        flow_tokens = None
        has_teacher = self.flow_enable and ("teacher_flow" in examples[0])
        if self.flow_enable:
            fh = self._run_flow_head(base_hidden, backbone_attention_mask)
            flow_tokens = fh["flow_tokens"]                          # [B,N,D]
            if has_teacher:
                F_teacher = torch.tensor(np.array([e["teacher_flow"] for e in examples]),
                                         device=device, dtype=torch.float32)   # [B,N,Hf,3]
                w = torch.tensor(np.array([e["teacher_weight"] for e in examples]),
                                 device=device, dtype=torch.float32)           # [B,N]
                fid_s = int(examples[0].get("frame_id", 0))
                fid_t = int(examples[0].get("frame_id", 0))
                flow_pred = fh["flow"].to(torch.float32)
                flow_loss = pw_distill_loss(flow_pred, F_teacher, w, fid_s, fid_t)

        # ---- action head (optionally conditioned on flow tokens) ----
        with torch.autocast("cuda", dtype=torch.float32):
            actions_t = torch.tensor(np.array(actions), device=device, dtype=dtype)
            actions_target = actions_t[:, -self.action_horizon:, :]
            rep = 2  # match QwenPI (no repeat for big FM)
            actions_target_repeated = actions_target.repeat(rep, 1, 1)

            vl_used, mask_used = vl_embs_list, backbone_attention_mask
            if self.flow_enable and self.flow_condition_action and flow_tokens is not None:
                vl_used, mask_used = self._augment_vl_with_flow(vl_embs_list, backbone_attention_mask, flow_tokens)

            vl_repeated = [h.repeat(rep, 1, 1) for h in vl_used]
            mask_repeated = mask_used.repeat(rep, 1).to(dtype=torch.bool) if mask_used is not None else None
            state_repeated = None
            if state is not None:
                st = torch.tensor(np.array(state), device=device, dtype=dtype)
                state_repeated = st.repeat(rep, 1, 1)

            action_loss = self.action_model(
                vl_repeated, actions_target_repeated, state_repeated,
                encoder_attention_mask=mask_repeated,
            )

        total = action_loss + (self.lambda_flow * flow_loss if has_teacher else 0.0)
        return {"loss": total, "action_loss": action_loss, "flow_loss": flow_loss}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs) -> np.ndarray:
        """RGB-only inference. Runs the flow head to produce flow_tokens (no teacher needed) and
        conditions the action head on them iff flow.condition_action — so deployment matches training."""
        from deployment.model_server.tools.image_tools import to_pil_preserve
        from starVLA.training.trainer_utils.trainer_tools import resize_images
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        vl_embs_list, backbone_attention_mask = self._encode_vl_hidden_states(batch_images, instructions)
        base_hidden = vl_embs_list[-1]
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)

        vl_used, mask_used = vl_embs_list, backbone_attention_mask
        if self.flow_enable and self.flow_condition_action:
            fh = self._run_flow_head(base_hidden, backbone_attention_mask)
            vl_used, mask_used = self._augment_vl_with_flow(vl_embs_list, backbone_attention_mask, fh["flow_tokens"])

        st = (torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=base_hidden.dtype)
              if state is not None else None)
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(vl_used, st, encoder_attention_mask=mask_used)
        return {"normalized_actions": pred_actions.detach().cpu().numpy()}
