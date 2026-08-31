from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import EventData, StaticGraph
from .model import HydroGraphOperator


@dataclass
class RolloutResult:
    node1: torch.Tensor
    node2: torch.Tensor
    edge1: torch.Tensor
    edge2: torch.Tensor
    start_t: int


def _static_col(graph: StaticGraph, node_type: int, names: list[str]) -> torch.Tensor | None:
    feats = graph.node1_feature_names if node_type == 1 else graph.node2_feature_names
    x = graph.node1_static if node_type == 1 else graph.node2_static
    for name in names:
        if name in feats:
            return x[:, feats.index(name)]
    return None


def sparse_initial_state(graph: StaticGraph, event: EventData, t: int) -> tuple[torch.Tensor, ...]:
    """Construct a no-future-truth hydraulic prior for sparse-observation assimilation."""
    n1 = torch.zeros_like(event.node1[t])
    n2 = torch.zeros_like(event.node2[t])
    e1 = torch.zeros_like(event.edge1[t])
    e2 = torch.zeros_like(event.edge2[t])
    base1 = _static_col(graph, 1, ["surface_elevation", "invert_elevation"])
    base2 = _static_col(graph, 2, ["elevation", "centroid_elevation", "min_elevation"])
    if base1 is not None:
        n1[:, 0] = base1
    if base2 is not None:
        n2[:, 1] = base2
    n2[:, 0] = event.node2[t, :, 0]  # rainfall is exogenous and known
    return n1, n2, e1, e2


def inject_observations(
    state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    event: EventData,
    t: int,
    mask1: torch.Tensor,
    mask2: torch.Tensor,
    observe_edges: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replace only declared sensor locations with truth; unobserved states remain model-generated."""
    n1, n2, e1, e2 = [x.clone() for x in state]
    # Sparse-city contract: sensors provide water level only. Auxiliary target variables
    # (1D inlet flow, 2D water volume, edge flow/velocity) remain model-generated.
    n1[mask1, 0] = event.node1[t, mask1, 0]
    n2[:, 0] = event.node2[t, :, 0]
    n2[mask2, 1] = event.node2[t, mask2, 1]
    if observe_edges:
        e1 = event.edge1[t].clone()
        e2 = event.edge2[t].clone()
    return n1, n2, e1, e2


def warm_hidden(
    model: HydroGraphOperator,
    graph: StaticGraph,
    event: EventData,
    end_t: int,
    context_steps: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Dense causal history assimilation; only timesteps <= end_t are consumed."""
    if end_t < 0:
        return None
    hidden = None
    start = max(0, end_t - context_steps + 1)
    ones1 = torch.ones(graph.n1, dtype=torch.bool, device=event.node1.device)
    ones2 = torch.ones(graph.n2, dtype=torch.bool, device=event.node2.device)
    for t in range(start, end_t + 1):
        next_rain = event.node2[min(t + 1, event.timesteps - 1), :, 0]
        pred = model.forward_step(graph, event.node1[t], event.node2[t], event.edge1[t], event.edge2[t],
                                  next_rain, ones1, ones2, hidden)
        hidden = (pred.hidden1, pred.hidden2)
    return hidden


def assimilate_sparse_history(
    model: HydroGraphOperator,
    graph: StaticGraph,
    event: EventData,
    end_t: int,
    context_steps: int,
    mask1: torch.Tensor,
    mask2: torch.Tensor,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor] | None]:
    """Causal sparse sensor assimilation without hidden-field leakage."""
    if end_t < 0:
        raise ValueError("end_t must be >=0 for sparse assimilation")
    start = max(0, end_t - context_steps + 1)
    state = sparse_initial_state(graph, event, start)
    state = inject_observations(state, event, start, mask1, mask2, observe_edges=False)
    hidden = None
    for t in range(start, end_t):
        n1, n2, e1, e2 = state
        pred = model.forward_step(graph, n1, n2, e1, e2, event.node2[t + 1, :, 0],
                                  mask1, mask2, hidden)
        hidden = (pred.hidden1, pred.hidden2)
        state = inject_observations((pred.node1, pred.node2, pred.edge1, pred.edge2),
                                    event, t + 1, mask1, mask2, observe_edges=False)
    return state, hidden


@torch.no_grad()
def rollout_event(
    model: HydroGraphOperator,
    graph: StaticGraph,
    event: EventData,
    warmup_steps: int = 10,
    context_steps: int = 4,
    obs_mask1: torch.Tensor | None = None,
    obs_mask2: torch.Tensor | None = None,
) -> RolloutResult:
    if event.timesteps <= warmup_steps:
        raise ValueError(f"Event {event.event_id} has only {event.timesteps} steps <= warmup {warmup_steps}")
    model.eval()
    t0 = warmup_steps - 1
    device = event.node1.device
    sparse = obs_mask1 is not None or obs_mask2 is not None
    if sparse:
        mask1 = torch.ones(graph.n1, dtype=torch.bool, device=device) if obs_mask1 is None else obs_mask1
        mask2 = torch.ones(graph.n2, dtype=torch.bool, device=device) if obs_mask2 is None else obs_mask2
        (cur1, cur2, cure1, cure2), hidden = assimilate_sparse_history(
            model, graph, event, t0, warmup_steps, mask1, mask2
        )
    else:
        mask1 = torch.ones(graph.n1, dtype=torch.bool, device=device)
        mask2 = torch.ones(graph.n2, dtype=torch.bool, device=device)
        hidden = warm_hidden(model, graph, event, t0 - 1, context_steps - 1)
        cur1, cur2, cure1, cure2 = event.node1[t0], event.node2[t0], event.edge1[t0], event.edge2[t0]

    out1, out2, oute1, oute2 = [], [], [], []
    for t in range(t0, event.timesteps - 1):
        pred = model.forward_step(graph, cur1, cur2, cure1, cure2, event.node2[t + 1, :, 0],
                                  mask1, mask2, hidden)
        out1.append(pred.node1)
        out2.append(pred.node2)
        oute1.append(pred.edge1)
        oute2.append(pred.edge2)
        # Forecast phase is fully autoregressive. Sensor observations after the warm-up are not
        # consumed unless a future online-assimilation protocol explicitly requests them.
        cur1, cur2, cure1, cure2 = pred.node1, pred.node2, pred.edge1, pred.edge2
        hidden = (pred.hidden1, pred.hidden2)
        mask1 = torch.zeros_like(mask1) if sparse else mask1
        mask2 = torch.zeros_like(mask2) if sparse else mask2
    return RolloutResult(torch.stack(out1), torch.stack(out2), torch.stack(oute1), torch.stack(oute2), start_t=t0 + 1)
