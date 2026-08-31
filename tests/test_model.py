import copy
import torch

from hydrograph.config import ModelConfig
from hydrograph.data import UrbanFloodBenchRepository
from hydrograph.model import HydroGraphOperator
from hydrograph.normalization import fit_normalization
from hydrograph.rollout import assimilate_sparse_history


def _model(repo, ref):
    g = repo.load_static(ref.model_id)
    e = repo.load_event(ref)
    st = fit_normalization([(g, e)])
    cfg = ModelConfig(hidden_dim=16, processor_layers=2, token_count=4, token_heads=2,
                      token_layers=1, dropout=0.0, adapter_rank=4)
    m = HydroGraphOperator(cfg, g.node1_static.shape[1], g.node2_static.shape[1],
                           g.edge1_static.shape[1], g.edge2_static.shape[1])
    m.set_normalization(st)
    m.eval()
    return g, e, m


def test_forward_shapes_and_finite(synthetic_root):
    repo = UrbanFloodBenchRepository(synthetic_root)
    ref = repo.list_events("train")[0]
    g, e, m = _model(repo, ref)
    p = m.forward_step(g, e.node1[0], e.node2[0], e.edge1[0], e.edge2[0], e.node2[1, :, 0])
    assert p.node1.shape == e.node1[0].shape
    assert p.node2.shape == e.node2[0].shape
    assert p.edge1.shape == e.edge1[0].shape
    assert p.edge2.shape == e.edge2[0].shape
    assert torch.isfinite(p.node2).all()
    assert (p.node2[:, 2] >= 0).all()


def test_sparse_assimilation_does_not_read_hidden_truth(synthetic_root):
    repo = UrbanFloodBenchRepository(synthetic_root)
    ref = repo.list_events("train")[0]
    g, e, m = _model(repo, ref)
    mask1 = torch.tensor([True, False, False])
    mask2 = torch.tensor([True] + [False] * (g.n2 - 1))
    state_a, _ = assimilate_sparse_history(m, g, e, end_t=5, context_steps=5, mask1=mask1, mask2=mask2)
    e2 = copy.deepcopy(e)
    e2.node1[:6, ~mask1] += 10000.0
    e2.node2[:6, ~mask2, 1:] += 10000.0
    e2.edge1[:6] += 10000.0
    e2.edge2[:6] += 10000.0
    state_b, _ = assimilate_sparse_history(m, g, e2, end_t=5, context_steps=5, mask1=mask1, mask2=mask2)
    for a, b in zip(state_a, state_b):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)
