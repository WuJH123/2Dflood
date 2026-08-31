from __future__ import annotations

import math
from collections import defaultdict

import torch

from .data import EventData, StaticGraph
from .physics import infer_dt_seconds, surface_mass_residual
from .rollout import RolloutResult

FT_TO_M = 0.3048


def _safe_float(x: torch.Tensor | float) -> float:
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().item()
    return float(x)


def regression_metrics(pred: torch.Tensor, true: torch.Tensor, prefix: str) -> dict[str, float]:
    pred, true = pred.float(), true.float()
    err = pred - true
    mse = torch.mean(err.square())
    rmse = torch.sqrt(mse)
    mae = torch.mean(err.abs())
    denom = torch.sum((true - true.mean()).square())
    nse = 1.0 - torch.sum(err.square()) / denom.clamp_min(1e-12)
    ss_tot = denom
    r2 = 1.0 - torch.sum(err.square()) / ss_tot.clamp_min(1e-12)
    return {f"{prefix}_rmse": _safe_float(rmse), f"{prefix}_mae": _safe_float(mae),
            f"{prefix}_nse": _safe_float(nse), f"{prefix}_r2": _safe_float(r2)}


def _surface_depth_ft(graph: StaticGraph, node2: torch.Tensor) -> torch.Tensor:
    if "area" not in graph.node2_feature_names:
        raise ValueError("Surface hazard metrics require 2D node area")
    area = graph.node2_static[:, graph.node2_feature_names.index("area")].clamp_min(1e-6)
    return node2[..., 2] / area


def _binary_scores(pred: torch.Tensor, true: torch.Tensor) -> dict[str, float]:
    tp = (pred & true).sum().float()
    fp = (pred & ~true).sum().float()
    fn = (~pred & true).sum().float()
    iou = tp / (tp + fp + fn).clamp_min(1)
    precision = tp / (tp + fp).clamp_min(1)
    recall = tp / (tp + fn).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {"iou": _safe_float(iou), "csi": _safe_float(iou), "f1": _safe_float(f1),
            "precision": _safe_float(precision), "recall": _safe_float(recall)}


def _first_true(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # mask [T,N]
    any_hit = mask.any(dim=0)
    idx = torch.argmax(mask.to(torch.int64), dim=0)
    return idx, any_hit


def event_metrics(
    graph: StaticGraph,
    event: EventData,
    result: RolloutResult,
    thresholds_m: list[float],
    arrival_threshold_m: float,
) -> dict[str, float]:
    s = result.start_t
    t1, t2, te1, te2 = event.node1[s:], event.node2[s:], event.edge1[s:], event.edge2[s:]
    p1, p2, pe1, pe2 = result.node1, result.node2, result.edge1, result.edge2
    if p1.shape != t1.shape or p2.shape != t2.shape:
        raise ValueError("Prediction/target shape mismatch in event_metrics")
    out: dict[str, float] = {}
    out.update(regression_metrics(p1[..., 0], t1[..., 0], "water_level_1d"))
    out.update(regression_metrics(p2[..., 1], t2[..., 1], "water_level_2d"))
    out.update(regression_metrics(p2[..., 2], t2[..., 2], "water_volume_2d"))
    out.update(regression_metrics(pe1[..., 0], te1[..., 0], "flow_1d"))
    out.update(regression_metrics(pe2[..., 0], te2[..., 0], "flow_2d"))
    out.update(regression_metrics(pe1[..., 1], te1[..., 1], "velocity_1d"))
    out.update(regression_metrics(pe2[..., 1], te2[..., 1], "velocity_2d"))

    pd = _surface_depth_ft(graph, p2) * FT_TO_M
    td = _surface_depth_ft(graph, t2) * FT_TO_M
    out.update(regression_metrics(pd, td, "surface_depth_m"))
    out["peak_depth_abs_error_m"] = _safe_float((pd.max() - td.max()).abs())
    out["peak_depth_rel_error"] = _safe_float((pd.max() - td.max()).abs() / td.max().abs().clamp_min(1e-6))
    for th in thresholds_m:
        scores = _binary_scores(pd >= th, td >= th)
        for k, v in scores.items():
            out[f"flood_{th:g}m_{k}"] = v

    th = arrival_threshold_m
    p_arr, p_any = _first_true(pd >= th)
    t_arr, t_any = _first_true(td >= th)
    common = t_any
    dt = infer_dt_seconds(event)
    if common.any():
        # Missing predicted arrival is penalized as the full forecast horizon.
        p_eff = torch.where(p_any, p_arr, torch.full_like(p_arr, pd.shape[0] - 1))
        out["arrival_time_mae_min"] = _safe_float((p_eff[common] - t_arr[common]).abs().float().mean() * dt / 60.0)
        p_dur = (pd >= th).sum(dim=0)
        t_dur = (td >= th).sum(dim=0)
        out["duration_mae_min"] = _safe_float((p_dur[common] - t_dur[common]).abs().float().mean() * dt / 60.0)
    else:
        out["arrival_time_mae_min"] = float("nan")
        out["duration_mae_min"] = float("nan")

    local, glob = [], []
    for k in range(p2.shape[0]):
        current = event.node2[s + k - 1] if k == 0 else p2[k - 1]
        mr = surface_mass_residual(graph, current, p1[k], p2[k], pe2[k], dt)
        local.append(mr.local_relative)
        glob.append(mr.global_relative)
    out["mass_local_relative"] = _safe_float(torch.stack(local).mean())
    out["mass_global_relative"] = _safe_float(torch.stack(glob).mean())
    out["nan_fraction"] = _safe_float(torch.isnan(p2).float().mean())
    out["negative_volume_fraction"] = _safe_float((p2[..., 2] < 0).float().mean())
    return out


def aggregate_event_balanced(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted(set().union(*(r.keys() for r in rows)))
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r and math.isfinite(r[k])]
        out[k] = sum(vals) / len(vals) if vals else float("nan")
    return out
