from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import torch

from .data import EventData, StaticGraph


@dataclass
class MassResidual:
    local_relative: torch.Tensor
    global_relative: torch.Tensor
    local_raw: torch.Tensor
    global_raw: torch.Tensor


def infer_dt_seconds(event: EventData, default: float = 60.0) -> float:
    if len(event.timestamps) < 2:
        return default
    a, b = event.timestamps[0], event.timestamps[1]
    for fmt in [None, "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"]:
        try:
            if fmt is None:
                da, db = datetime.fromisoformat(a), datetime.fromisoformat(b)
            else:
                da, db = datetime.strptime(a, fmt), datetime.strptime(b, fmt)
            dt = (db - da).total_seconds()
            if dt > 0:
                return float(dt)
        except (ValueError, TypeError):
            continue
    return default


def _feature(graph: StaticGraph, name: str, fallback: float = 1.0) -> torch.Tensor:
    if name in graph.node2_feature_names:
        return graph.node2_static[:, graph.node2_feature_names.index(name)]
    return torch.full((graph.n2,), fallback, dtype=graph.node2_static.dtype, device=graph.node2_static.device)


def surface_mass_residual(
    graph: StaticGraph,
    current_node2: torch.Tensor,
    next_node1: torch.Tensor,
    next_node2: torch.Tensor,
    next_edge2: torch.Tensor,
    dt_seconds: float,
) -> MassResidual:
    """Mass residual for the 2D surface domain in native UrbanFloodBench imperial units.

    Rainfall is interval depth [in], area [ft^2], volume [ft^3], edge/inlet flow [ft^3/s].
    The released ML-ready data supplies aggregate 1D-node inlet flow rather than a per-coupling
    exchange flow. When a 1D node has multiple 2D connections, its exchange is distributed equally
    across those connections for *local* diagnostics. The global residual uses the exact aggregate.
    """
    area = _feature(graph, "area", 1.0).to(next_node2.device)
    rain_ft = next_node2[:, 0] / 12.0
    rain_volume = rain_ft * area
    delta_volume = next_node2[:, 2] - current_node2[:, 2]

    flow = next_edge2[:, 0]
    src, dst = graph.edge2_index
    net = torch.zeros(graph.n2, device=flow.device, dtype=flow.dtype)
    net.index_add_(0, dst, flow)
    net.index_add_(0, src, -flow)
    surface_flow_volume = net * dt_seconds

    c1, c2 = graph.coupling_index
    inlet = next_node1[:, 1]
    counts = torch.bincount(c1, minlength=graph.n1).to(inlet.device).clamp_min(1)
    share = inlet[c1] / counts[c1]
    coupling = torch.zeros(graph.n2, device=inlet.device, dtype=inlet.dtype)
    coupling.index_add_(0, c2, share)
    coupling_volume = coupling * dt_seconds

    expected = rain_volume + surface_flow_volume - coupling_volume
    raw = delta_volume - expected
    scale = delta_volume.abs() + rain_volume.abs() + surface_flow_volume.abs() + coupling_volume.abs() + 1.0
    local_rel = raw.abs() / scale

    global_raw = raw.sum().abs()
    global_scale = delta_volume.abs().sum() + rain_volume.abs().sum() + coupling_volume.abs().sum() + 1.0
    global_rel = global_raw / global_scale
    return MassResidual(local_rel.mean(), global_rel, raw, global_raw)
