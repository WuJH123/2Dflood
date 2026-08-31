from __future__ import annotations

import torch

from .data import EventData
from .rollout import RolloutResult


@torch.no_grad()
def persistence_rollout(event: EventData, warmup_steps: int = 10) -> RolloutResult:
    if event.timesteps <= warmup_steps:
        raise ValueError("Event too short for persistence rollout")
    t0 = warmup_steps - 1
    n1 = event.node1[t0].clone()
    n2 = event.node2[t0].clone()
    e1 = event.edge1[t0].clone()
    e2 = event.edge2[t0].clone()
    out1, out2, out_e1, out_e2 = [], [], [], []
    for t in range(t0 + 1, event.timesteps):
        n2 = n2.clone()
        n2[:, 0] = event.node2[t, :, 0]  # known rainfall forcing, hydraulic state persistence
        out1.append(n1.clone())
        out2.append(n2.clone())
        out_e1.append(e1.clone())
        out_e2.append(e2.clone())
    return RolloutResult(torch.stack(out1), torch.stack(out2), torch.stack(out_e1), torch.stack(out_e2), t0 + 1)
