from __future__ import annotations

from collections import defaultdict

import torch

from .data import EventData, StaticGraph
from .model import HydroGraphOperator

FT_TO_M = 0.3048


def _bin(values: torch.Tensor, edges: list[float]) -> torch.Tensor:
    bounds = torch.tensor(edges, device=values.device, dtype=values.dtype)
    return torch.bucketize(values, bounds)


@torch.no_grad()
def hydraulic_regime_embeddings(
    model: HydroGraphOperator,
    graph: StaticGraph,
    event: EventData,
    timestep_stride: int = 2,
    depth_edges_m: list[float] | None = None,
    rain_edges_in: list[float] | None = None,
) -> dict[tuple[int, int], torch.Tensor]:
    """Aggregate latent surface representations by matched physical regimes.

    Regime matching provides a defensible cross-city correspondence for representation analysis:
    cities are compared at similar depth/rainfall states rather than arbitrary node identities.
    """
    depth_edges_m = depth_edges_m or [0.01, 0.05, 0.10, 0.30, 0.50]
    rain_edges_in = rain_edges_in or [0.001, 0.01, 0.05, 0.10]
    if "area" not in graph.node2_feature_names:
        raise ValueError("Hydraulic regime analysis requires 2D node area")
    area = graph.node2_static[:, graph.node2_feature_names.index("area")].clamp_min(1e-6)
    sums: dict[tuple[int, int], torch.Tensor] = {}
    counts: dict[tuple[int, int], int] = defaultdict(int)
    hidden = None
    ones1 = torch.ones(graph.n1, dtype=torch.bool, device=event.node1.device)
    ones2 = torch.ones(graph.n2, dtype=torch.bool, device=event.node2.device)
    for t in range(0, event.timesteps - 1, max(1, timestep_stride)):
        pred = model.forward_step(graph, event.node1[t], event.node2[t], event.edge1[t], event.edge2[t],
                                  event.node2[t, :, 0], ones1, ones2, hidden)
        hidden = (pred.hidden1, pred.hidden2)
        depth_m = event.node2[t, :, 2] / area * FT_TO_M
        rain = event.node2[t, :, 0]
        db = _bin(depth_m, depth_edges_m)
        rb = _bin(rain, rain_edges_in)
        for d in torch.unique(db):
            for r in torch.unique(rb):
                mask = (db == d) & (rb == r)
                if not mask.any():
                    continue
                key = (int(d), int(r))
                x = pred.hidden2[mask].sum(dim=0)
                sums[key] = x if key not in sums else sums[key] + x
                counts[key] += int(mask.sum())
    return {k: v / counts[k] for k, v in sums.items() if counts[k] > 0}


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("CKA requires matched [regime, feature] matrices")
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    xy = (x.T @ y).square().sum()
    xx = (x.T @ x).square().sum().sqrt()
    yy = (y.T @ y).square().sum().sqrt()
    return float((xy / (xx * yy).clamp_min(1e-12)).cpu())


def matched_regime_cka(a: dict[tuple[int, int], torch.Tensor],
                       b: dict[tuple[int, int], torch.Tensor], min_regimes: int = 4) -> dict[str, float | int]:
    common = sorted(set(a) & set(b))
    if len(common) < min_regimes:
        return {"cka": float("nan"), "matched_regimes": len(common)}
    x = torch.stack([a[k] for k in common])
    y = torch.stack([b[k] for k in common])
    return {"cka": linear_cka(x, y), "matched_regimes": len(common)}
