# tests/geomemvla/test_pw_probe_metrics.py
# [LangPointWorld] Unit tests for probe gate metrics (pure, CPU).
import torch
from starVLA.model.modules.langpw.probe_metrics import keypoint_corr, filtered_l2, gate_verdict


def test_perfect_prediction_corr_is_one_and_l2_zero():
    torch.manual_seed(0)
    gt = torch.randn(16, 4, 3)
    mask = torch.ones(16, dtype=torch.bool)
    c = keypoint_corr(gt.clone(), gt.clone(), mask)
    assert c["corr_mean"] > 0.99
    l = filtered_l2(gt.clone(), gt.clone(), mask)
    assert l["l2_mean"] < 1e-5


def test_anti_correlated_fails_gate():
    torch.manual_seed(1)
    gt = torch.randn(16, 4, 3)
    mask = torch.ones(16, dtype=torch.bool)
    c = keypoint_corr(-gt, gt, mask)   # sign-flipped = opposite direction / under-drive
    assert c["corr_mean"] < 0.0
    assert gate_verdict(c["corr_mean"]) is False


def test_mask_excludes_points():
    torch.manual_seed(2)
    gt = torch.randn(8, 2, 3)
    pred = gt.clone()
    pred[4:] = 0.0
    mask = torch.zeros(8, dtype=torch.bool)
    mask[:4] = True   # only first 4 are perfect
    c = keypoint_corr(pred, gt, mask)
    assert c["corr_mean"] > 0.99


def test_gate_threshold():
    assert gate_verdict(0.71) is True
    assert gate_verdict(0.69) is False
