# tests/geomemvla/test_d4rt_probe.py
# [D4RT-WorldState] Query-builder contract for the forecasting probe (no model).
import torch


def test_build_query_pins_tcam_to_ttgt():
    from tools.d4rt_forecast_probe import build_forecast_query
    M = 5
    uv = torch.rand(M, 2)
    q = build_forecast_query(uv, t_src=0, t_tgt=7, device="cpu")
    assert set(q.keys()) == {"u", "v", "t_src", "t_tgt", "t_cam"}
    assert torch.equal(q["t_cam"], q["t_tgt"])          # t_cam == t_tgt invariant (spec R2)
    assert q["u"].shape == (1, M) and q["t_tgt"].dtype == torch.long
    assert q["t_src"].dtype == torch.long and q["t_cam"].dtype == torch.long
