# [LangPointWorld] Hybrid sparse point selection (spec §5.3), CPU.
import torch
from starVLA.model.modules.langpw.point_sampling import sample_points


def test_allocation_and_weights():
    torch.manual_seed(0)
    Ns = 2000
    Hf = 4
    coord = torch.rand(Ns, 3)
    flow = torch.rand(Hf, Ns, 3) * 0.01
    obj = torch.zeros(Ns, dtype=torch.bool); obj[:500] = True
    tgt = torch.zeros(Ns, dtype=torch.bool); tgt[500:900] = True
    grip = torch.zeros(Ns, dtype=torch.bool); grip[900:1100] = True
    out = sample_points(coord, flow, obj, tgt, grip, n_points=256, alloc=(96, 48, 48, 64))
    assert out["indices"].shape == (256,)
    assert out["role"].shape == (256,) and out["weight"].shape == (256,)
    w, role = out["weight"], out["role"]
    assert w[role == 0].mean() > w[role == 3].mean()  # object weighted above background


def test_handles_missing_masks_gracefully():
    torch.manual_seed(1)
    Ns = 500
    out = sample_points(torch.rand(Ns, 3), torch.rand(2, Ns, 3),
                        torch.zeros(Ns, dtype=torch.bool), torch.zeros(Ns, dtype=torch.bool),
                        torch.zeros(Ns, dtype=torch.bool), n_points=128, alloc=(32, 32, 32, 32))
    assert out["indices"].shape == (128,)


def test_fixed_n_even_for_tiny_scene():
    torch.manual_seed(2)
    Ns = 40  # smaller than n_points -> must still return exactly n_points (with repeats)
    out = sample_points(torch.rand(Ns, 3), torch.rand(2, Ns, 3),
                        torch.zeros(Ns, dtype=torch.bool), torch.zeros(Ns, dtype=torch.bool),
                        torch.zeros(Ns, dtype=torch.bool), n_points=128, alloc=(32, 32, 32, 32))
    assert out["indices"].shape == (128,)
