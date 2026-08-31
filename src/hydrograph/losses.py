from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import LossConfig
from .data import StaticGraph
from .model import StepPrediction
from .physics import surface_mass_residual


@dataclass
class LossBreakdown:
    total: torch.Tensor
    terms: dict[str, torch.Tensor]


def _rmse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(F.mse_loss(a, b) + 1e-8)


def compute_step_loss(
    pred: StepPrediction,
    target1: torch.Tensor,
    target2: torch.Tensor,
    target_e1: torch.Tensor,
    target_e2: torch.Tensor,
    current2: torch.Tensor,
    graph: StaticGraph,
    cfg: LossConfig,
    dt_seconds: float,
) -> LossBreakdown:
    terms = {
        "water_level_1d": _rmse(pred.node1[:, 0], target1[:, 0]),
        "inlet_flow": _rmse(pred.node1[:, 1], target1[:, 1]),
        "water_level_2d": _rmse(pred.node2[:, 1], target2[:, 1]),
        "water_volume": _rmse(pred.node2[:, 2], target2[:, 2]),
        "edge_flow_1d": _rmse(pred.edge1[:, 0], target_e1[:, 0]),
        "edge_velocity_1d": _rmse(pred.edge1[:, 1], target_e1[:, 1]),
        "edge_flow_2d": _rmse(pred.edge2[:, 0], target_e2[:, 0]),
        "edge_velocity_2d": _rmse(pred.edge2[:, 1], target_e2[:, 1]),
    }
    mass = surface_mass_residual(graph, current2, pred.node1, pred.node2, pred.edge2, dt_seconds)
    terms["mass_local"] = mass.local_relative
    terms["mass_global"] = mass.global_relative
    terms["nonnegative_volume"] = torch.relu(-pred.node2[:, 2]).mean()
    total = (
        cfg.water_level * 0.5 * (terms["water_level_1d"] + terms["water_level_2d"])
        + cfg.water_volume * terms["water_volume"]
        + cfg.inlet_flow * terms["inlet_flow"]
        + cfg.edge_flow_1d * terms["edge_flow_1d"]
        + cfg.edge_flow_2d * terms["edge_flow_2d"]
        + cfg.edge_velocity_1d * terms["edge_velocity_1d"]
        + cfg.edge_velocity_2d * terms["edge_velocity_2d"]
        + cfg.mass_local * terms["mass_local"]
        + cfg.mass_global * terms["mass_global"]
        + cfg.dry_nonnegative * terms["nonnegative_volume"]
    )
    return LossBreakdown(total, terms)
