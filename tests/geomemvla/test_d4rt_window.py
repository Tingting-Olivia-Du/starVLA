# tests/geomemvla/test_d4rt_window.py
# [D4RT-WorldState] Agentview temporal window builder contract.
import torch
from PIL import Image


def _ex(n_views, with_window=False, win_frames=4):
    img = Image.new("RGB", (256, 256))
    e = {"image": [img for _ in range(n_views)]}
    if with_window:
        # image_window is List[F][views]; agentview = view 0 of each frame.
        e["image_window"] = [[img for _ in range(n_views)] for _ in range(win_frames)]
    return e


def test_window_is_agentview_only_and_normalized():
    from starVLA.model.framework.VLM4A.GeoMemoryVLA import _build_d4rt_window
    examples = [_ex(2), _ex(2)]      # 2 cameras present; D4RT must take only agentview (view 0)
    win = _build_d4rt_window(examples, device="cpu", win_len=4)
    assert win.shape == (2, 4, 3, 256, 256)
    assert win.min() >= 0.0 and win.max() <= 1.0


def test_window_uses_image_window_when_present():
    from starVLA.model.framework.VLM4A.GeoMemoryVLA import _build_d4rt_window
    examples = [_ex(2, with_window=True, win_frames=6)]
    win = _build_d4rt_window(examples, device="cpu", win_len=6)
    assert win.shape == (1, 6, 3, 256, 256)
