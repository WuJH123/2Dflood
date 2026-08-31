from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .data import EventData, StaticGraph


@dataclass
class TensorStats:
    mean: torch.Tensor
    std: torch.Tensor

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def denorm(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x.device) + self.mean.to(x.device)


@dataclass
class NormalizationBundle:
    node1_static: TensorStats
    node2_static: TensorStats
    edge1_static: TensorStats
    edge2_static: TensorStats
    node1_dynamic: TensorStats
    node2_dynamic: TensorStats
    edge1_dynamic: TensorStats
    edge2_dynamic: TensorStats
    fitted_models: tuple[str, ...]
    fitted_events: tuple[str, ...]

    def state_dict(self) -> dict:
        return {
            k: {"mean": getattr(self, k).mean.cpu(), "std": getattr(self, k).std.cpu()}
            for k in ["node1_static", "node2_static", "edge1_static", "edge2_static", "node1_dynamic", "node2_dynamic", "edge1_dynamic", "edge2_dynamic"]
        } | {"fitted_models": self.fitted_models, "fitted_events": self.fitted_events}

    @classmethod
    def from_state_dict(cls, d: dict) -> "NormalizationBundle":
        kwargs = {k: TensorStats(d[k]["mean"], d[k]["std"]) for k in ["node1_static", "node2_static", "edge1_static", "edge2_static", "node1_dynamic", "node2_dynamic", "edge1_dynamic", "edge2_dynamic"]}
        return cls(**kwargs, fitted_models=tuple(d["fitted_models"]), fitted_events=tuple(d["fitted_events"]))


class StreamingMoments:
    def __init__(self, dim: int):
        self.dim = dim
        self.sum = torch.zeros(dim, dtype=torch.float64)
        self.sumsq = torch.zeros(dim, dtype=torch.float64)
        self.count = 0

    def update(self, chunk: torch.Tensor) -> None:
        if self.dim == 0 or chunk.numel() == 0:
            return
        x = chunk.detach().cpu().reshape(-1, self.dim).double()
        finite = torch.isfinite(x)
        counts = finite.sum(dim=0)
        if not torch.equal(counts, counts[:1].expand_as(counts)):
            raise ValueError("Feature columns contain inconsistent non-finite coverage")
        x = torch.where(finite, x, torch.zeros_like(x))
        self.sum += x.sum(dim=0)
        self.sumsq += x.square().sum(dim=0)
        self.count += int(counts[0])

    def finish(self) -> TensorStats:
        if self.dim == 0:
            return TensorStats(torch.zeros(0), torch.ones(0))
        if self.count == 0:
            return TensorStats(torch.zeros(self.dim), torch.ones(self.dim))
        mean = self.sum / self.count
        var = (self.sumsq / self.count - mean.square()).clamp_min(0)
        return TensorStats(mean.float(), torch.sqrt(var).float().clamp_min(1e-6))


def fit_normalization(items: Iterable[tuple[StaticGraph, EventData]]) -> NormalizationBundle:
    """One-pass, memory-safe source-training statistics.

    The caller must supply *only* training events. This design makes the leakage boundary explicit.
    Static features are counted once per model, while dynamic features are counted once per event.
    """
    moments: dict[str, StreamingMoments] | None = None
    seen_models: dict[str, StaticGraph] = {}
    models: set[str] = set()
    events: list[str] = []
    for g, e in items:
        if moments is None:
            moments = {
                "node1_static": StreamingMoments(g.node1_static.shape[-1]),
                "node2_static": StreamingMoments(g.node2_static.shape[-1]),
                "edge1_static": StreamingMoments(g.edge1_static.shape[-1]),
                "edge2_static": StreamingMoments(g.edge2_static.shape[-1]),
                "node1_dynamic": StreamingMoments(e.node1.shape[-1]),
                "node2_dynamic": StreamingMoments(e.node2.shape[-1]),
                "edge1_dynamic": StreamingMoments(e.edge1.shape[-1]),
                "edge2_dynamic": StreamingMoments(e.edge2.shape[-1]),
            }
        expected = {
            "node1_static": g.node1_static.shape[-1], "node2_static": g.node2_static.shape[-1],
            "edge1_static": g.edge1_static.shape[-1], "edge2_static": g.edge2_static.shape[-1],
            "node1_dynamic": e.node1.shape[-1], "node2_dynamic": e.node2.shape[-1],
            "edge1_dynamic": e.edge1.shape[-1], "edge2_dynamic": e.edge2.shape[-1],
        }
        for k, dim in expected.items():
            if moments[k].dim != dim:
                raise ValueError(f"Cross-city feature dimension mismatch for {k}: {moments[k].dim} vs {dim}")
        if g.model_id not in seen_models:
            moments["node1_static"].update(g.node1_static)
            moments["node2_static"].update(g.node2_static)
            moments["edge1_static"].update(g.edge1_static)
            moments["edge2_static"].update(g.edge2_static)
            seen_models[g.model_id] = g
        moments["node1_dynamic"].update(e.node1)
        moments["node2_dynamic"].update(e.node2)
        moments["edge1_dynamic"].update(e.edge1)
        moments["edge2_dynamic"].update(e.edge2)
        models.add(g.model_id)
        events.append(f"{e.model_id}/{e.event_id}")
    if moments is None:
        raise ValueError("Cannot fit normalization on empty training set")
    return NormalizationBundle(
        **{k: v.finish() for k, v in moments.items()},
        fitted_models=tuple(sorted(models)), fitted_events=tuple(sorted(events)),
    )
