# tests/geomemvla/test_pw_ee_frame.py
# [LangPointWorld] Unit tests for the EE-centric frame transform (pure, CPU).
import torch, pytest
from starVLA.model.modules.langpw.ee_frame import to_ee_frame, flow_to_ee_frame, assert_same_frame


def _quat_identity():
    return torch.tensor([0.0, 0.0, 0.0, 1.0])  # (x,y,z,w)


def test_identity_ee_pose_is_translation_only():
    pts = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    ee_pos = torch.tensor([1.0, 1.0, 1.0])
    out = to_ee_frame(pts, ee_pos, _quat_identity())
    assert torch.allclose(out, pts - ee_pos, atol=1e-6)


def test_flow_is_rotation_only_not_translated():
    # a pure displacement must be invariant to EE translation
    flow = torch.tensor([[0.5, 0.0, 0.0]])
    out = flow_to_ee_frame(flow, _quat_identity())
    assert torch.allclose(out, flow, atol=1e-6)


def test_90deg_z_rotation_maps_x_to_minus_y():
    # quat for +90deg about z: (0,0,sin45,cos45). world->ee applies the INVERSE (conjugate)
    # rotation, so a world +x vector expressed in a frame rotated +90 about z lands at -y.
    # (This is the standard world->body transform: R(-90z) @ [1,0,0] = [0,-1,0].)
    s = (2 ** 0.5) / 2
    quat = torch.tensor([0.0, 0.0, s, s])
    flow = torch.tensor([[1.0, 0.0, 0.0]])
    out = flow_to_ee_frame(flow, quat)
    assert torch.allclose(out, torch.tensor([[0.0, -1.0, 0.0]]), atol=1e-5)


def test_assert_same_frame_rejects_mismatch():
    assert_same_frame(5, 5)  # ok
    with pytest.raises(ValueError):
        assert_same_frame(5, 6)
