# [LangPointWorld] Distillation loss tests (pure, CPU).
import torch, pytest
from starVLA.model.modules.langpw.pw_distill_loss import pw_distill_loss


def test_zero_when_equal():
    torch.manual_seed(0)
    f = torch.randn(2, 8, 4, 3)
    w = torch.ones(2, 8)
    assert float(pw_distill_loss(f, f.clone(), w, 0, 0)) < 1e-6


def test_weight_scales_and_frame_guard():
    fs = torch.zeros(1, 2, 1, 3)
    ft = torch.ones(1, 2, 1, 3)
    w = torch.tensor([[2.0, 0.0]])   # only the first point contributes
    loss = pw_distill_loss(fs, ft, w, 3, 3)
    assert loss > 0
    with pytest.raises(ValueError):
        pw_distill_loss(fs, ft, w, 3, 4)   # frame mismatch


def test_zero_weight_point_ignored():
    # a huge error on a zero-weight point must not affect the loss
    fs = torch.zeros(1, 2, 1, 3)
    ft = torch.zeros(1, 2, 1, 3)
    ft[0, 1] = 100.0
    w = torch.tensor([[1.0, 0.0]])
    assert float(pw_distill_loss(fs, ft, w, 0, 0)) < 1e-6
