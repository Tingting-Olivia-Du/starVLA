# tests/geomemvla/test_d4rt_loader.py
# [D4RT-WorldState] Loader import + vendored D4RTModel construction from a model.yaml
# (no checkpoint download).
import glob
import pytest


def test_loader_imports():
    from starVLA.model.modules.d4rt.loader import build_d4rt_model
    assert callable(build_d4rt_model)


def test_builds_model_from_yaml_without_ckpt():
    # Build the architecture from a vendored model.yaml; no weights needed.
    from starVLA.model.modules.d4rt.loader import build_d4rt_model
    yamls = glob.glob("starVLA/model/modules/d4rt/**/model_yaml*.yaml", recursive=True)
    if not yamls:
        pytest.skip("no vendored model.yaml present")
    model = build_d4rt_model(yamls[0], ckpt_path=None)
    assert hasattr(model, "encode_video") and hasattr(model, "decode_queries")
