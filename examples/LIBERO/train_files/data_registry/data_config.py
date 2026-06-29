"""LIBERO benchmark — data config, embodiment tags, and mixtures."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------
class Libero4in1DataConfig:
    embodiment_tag = EmbodimentTag.FRANKA
    video_keys = [
        "video.primary_image",
        "video.wrist_image",
    ]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.pad",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))
    state_indices = [0]

    # [Geo-MemoryVLA] image_window: multi-frame window for VGGTWorldModel imagination.
    # Defaults follow VGGT-World (arXiv 2603.12655): context k=2, chunk m=2 -> 5 frames.
    # Window length = context + chunk + 1 (Stage-2's required_frames). See spec.
    enable_image_window = True
    image_window_context = 2
    image_window_chunk = 2

    # [D4RT-WorldState] D4RT is a monocular reconstructor over a contiguous clip — it wants a
    # window of `d4rt_window_frames` PAST frames ending at the current frame (TIME axis, primary
    # camera only, B1), NOT VGGT's context/chunk window. When >0, this overrides the index range.
    # Set by the dataloader gate from framework.world_state.clip_frames when backbone==d4rt.
    d4rt_window_frames = 0

    @property
    def image_window_indices(self):
        if self.d4rt_window_frames and self.d4rt_window_frames > 0:
            # D4RT: clip_frames contiguous past frames ending at current, e.g. 48 -> [-47..0].
            return list(range(-(self.d4rt_window_frames - 1), 1))
        # VGGT: range(-(ctx-1), chunk+2): 1 past + current + chunk+1 future = ctx+chunk+1 frames.
        return list(range(-(self.image_window_context - 1), self.image_window_chunk + 2))

    def modality_config(self):
        configs = {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys),  # ignore state modality for now since some datasets don't have state and we want to be able to use them, can add back later if needed
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }
        # [Geo-MemoryVLA] Declare the window modality only when enabled (imagination on).
        # Omitting it => no extra frame reads and no image_window key in the sample.
        # S1 fix: image_window is a SINGLE-CAMERA temporal sequence (primary only) — VGGT-World
        # is a monocular-trajectory world model, so its frame axis must be TIME, not views.
        # Using all video_keys interleaved views into the time axis (views mistaken for
        # timesteps). Multi-view imagination is a deferred extension (VGGT supports multi-view).
        if self.enable_image_window:
            configs["image_window"] = ModalityConfig(
                delta_indices=self.image_window_indices,
                modality_keys=[self.video_keys[0]],  # primary camera only -> frames == timesteps
            )
        return configs

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x": "min_max",
                    "action.y": "min_max",
                    "action.z": "min_max",
                    "action.roll": "min_max",
                    "action.pitch": "min_max",
                    "action.yaw": "min_max",
                },
            ),
        ])


ROBOT_TYPE_CONFIG_MAP = {
    "libero_franka": Libero4in1DataConfig(),
}


# ---------------------------------------------------------------------------
# Embodiment Tags
# ---------------------------------------------------------------------------
ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    # Per Proposal A, embodiment_tag now lives as a classvar on each DataConfig.
    # The registry derives ROBOT_TYPE_TO_EMBODIMENT_TAG automatically. Kept as
    # an empty dict for backward compat (it is honored as legacy override).
}


# ---------------------------------------------------------------------------
# Mixtures
# ---------------------------------------------------------------------------
DATASET_NAMED_MIXTURES = {
    "libero_all": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_goal": [
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "multi_robot": [
        ("LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
}
