# tests/geomemvla/test_imagination_adapter.py
# [Geo-MemoryVLA] The adapter wraps the vendored VGGTWorldModel; here we only assert
# the adapter module imports and the vendored loss contract is intact (no VGGT download).
import torch


def test_world_model_loss_contract():
    # Vendored WorldModelLoss returns an 'objective' scalar from latent tokens.
    from starVLA.model.modules.vggt_world.losses import WorldModelLoss

    loss = WorldModelLoss(latent_weight=1.0)
    preds = {
        "pred_state_tokens": torch.randn(2, 10, 1024),
        "target_state_tokens": torch.randn(2, 10, 1024),
        "stage": "stage2",
    }
    out = loss(preds, batch={})
    assert "objective" in out and out["objective"].dim() == 0


def test_adapter_imports():
    from starVLA.model.modules.geomem.imagination_adapter import ImaginationAdapter
    assert hasattr(ImaginationAdapter, "training_loss")
    assert hasattr(ImaginationAdapter, "imagine_tokens")
