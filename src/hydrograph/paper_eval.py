from __future__ import annotations

import json
from pathlib import Path

import torch

from .adapt import adapt_sparse_city, evaluate_sparse_adapted, select_sensor_masks
from .baselines import persistence_rollout
from .config import ExperimentConfig
from .data import EventRef, UrbanFloodBenchRepository
from .metrics import aggregate_event_balanced, event_metrics
from .rollout import rollout_event
from .split import few_shot_target_split
from .trainer import load_model_checkpoint
from .utils import atomic_json_dump, choose_device


def summarize_replicates(metrics_by_seed: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    if not metrics_by_seed:
        return {}
    keys = sorted(set().union(*(m.keys() for m in metrics_by_seed.values())))
    out = {}
    for k in keys:
        vals = [m[k] for m in metrics_by_seed.values() if k in m and isinstance(m[k], float)]
        vals = [v for v in vals if v == v and abs(v) != float("inf")]
        if vals:
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            out[k] = {"mean": mean, "std": std, "n": len(vals)}
    return out


def transfer_recovery(error_zero_sparse: float, error_adapted: float, error_dense: float) -> float:
    """Fraction of the sparse-to-dense error gap recovered by adaptation; 1 is dense parity."""
    denom = error_zero_sparse - error_dense
    if abs(denom) < 1e-12:
        return float("nan")
    return (error_zero_sparse - error_adapted) / denom


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint: str | Path,
    cfg: ExperimentConfig,
    repo: UrbanFloodBenchRepository,
    refs: list[EventRef],
    output_dir: str | Path | None = None,
    sparse_fraction: float | None = None,
    sensor_seed: int | None = None,
) -> dict[str, float]:
    if not refs:
        return {}
    device = choose_device(cfg.train.device)
    graphs = {}
    models = {}
    masks = {}
    rows = []
    for ref in refs:
        if ref.model_id not in graphs:
            g0 = repo.load_static(ref.model_id, ref.split)
            graphs[ref.model_id] = g0.to(device)
            models[ref.model_id], _ = load_model_checkpoint(checkpoint, device, g0)
            if sparse_fraction is not None:
                m1, m2 = select_sensor_masks(g0, sparse_fraction, sensor_seed or cfg.train.seed)
                masks[ref.model_id] = (m1.to(device), m2.to(device))
        graph, model = graphs[ref.model_id], models[ref.model_id]
        event = repo.load_event(ref).to(device)
        if sparse_fraction is None:
            result = rollout_event(model, graph, event, cfg.data.warmup_steps, cfg.data.context_steps)
        else:
            m1, m2 = masks[ref.model_id]
            result = rollout_event(model, graph, event, cfg.data.warmup_steps, cfg.data.context_steps, m1, m2)
        metrics = event_metrics(graph, event, result, cfg.evaluation.flood_thresholds_m,
                                cfg.evaluation.arrival_threshold_m)
        rows.append({"model_id": ref.model_id, "event_id": ref.event_id, **metrics})
        if output_dir is not None and cfg.evaluation.save_predictions:
            pred_dir = Path(output_dir) / "predictions"
            pred_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "contract": "HYDROGRAPH_PREDICTION_V1", "model_id": ref.model_id,
                "event_id": ref.event_id, "start_t": result.start_t,
                "node1": result.node1.cpu(), "node2": result.node2.cpu(),
                "edge1": result.edge1.cpu(), "edge2": result.edge2.cpu(),
            }, pred_dir / f"{ref.model_id}__{ref.event_id}.pt")
    agg = aggregate_event_balanced([{k: v for k, v in r.items() if isinstance(v, float)} for r in rows])
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        atomic_json_dump(rows, out / "event_metrics.json")
        atomic_json_dump(agg, out / "event_balanced_metrics.json")
    return agg


@torch.no_grad()
def evaluate_persistence(
    cfg: ExperimentConfig,
    repo: UrbanFloodBenchRepository,
    refs: list[EventRef],
) -> dict[str, float]:
    rows = []
    for ref in refs:
        graph = repo.load_static(ref.model_id, ref.split)
        event = repo.load_event(ref)
        result = persistence_rollout(event, cfg.data.warmup_steps)
        rows.append(event_metrics(graph, event, result, cfg.evaluation.flood_thresholds_m,
                                  cfg.evaluation.arrival_threshold_m))
    return aggregate_event_balanced(rows)


def run_target_protocol(
    checkpoint: str | Path,
    cfg: ExperimentConfig,
    repo: UrbanFloodBenchRepository,
    target_refs: list[EventRef],
    output_dir: str | Path,
    adaptation_epochs: int = 20,
) -> dict:
    """Generate the primary Water Research evidence matrix for one held-out city.

    All target adaptation/evaluation partitions are event-disjoint. Sparse adaptation consumes only
    target-city sensor water levels; full target fields remain evaluation-only.
    """
    if not target_refs:
        raise ValueError("No target events")
    target = target_refs[0].model_id
    if any(r.model_id != target for r in target_refs):
        raise ValueError("run_target_protocol expects refs from exactly one target city")
    ck_meta = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ck_manifest = ck_meta.get("manifest", {})
    if ck_manifest.get("protocol") != "leave_one_city_out":
        raise RuntimeError("Paper target protocol requires a leave_one_city_out source checkpoint")
    if ck_manifest.get("target_model") != target:
        raise RuntimeError(f"Checkpoint held out {ck_manifest.get('target_model')}, not requested target {target}")
    if target in set(ck_meta.get("normalization", {}).get("fitted_models", [])):
        raise RuntimeError("P0 leakage: target city appears in source normalization lineage")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"target_model": target, "protocol": "PAPER_TARGET_PROTOCOL_V1"}
    report["persistence"] = evaluate_persistence(cfg, repo, target_refs)
    report["zero_shot_dense_warmup"] = evaluate_checkpoint(
        checkpoint, cfg, repo, target_refs, out / "zero_shot_dense"
    )
    sparse_zero = {}
    for frac in cfg.evaluation.sparse_sensor_fractions:
        reps = {}
        for seed in cfg.evaluation.sensor_layout_seeds:
            reps[str(seed)] = evaluate_checkpoint(
                checkpoint, cfg, repo, target_refs, out / f"zero_shot_sparse_{frac:g}_seed{seed}",
                sparse_fraction=frac, sensor_seed=seed,
            )
        sparse_zero[str(frac)] = {"replicates": reps, "summary": summarize_replicates(reps)}
    report["zero_shot_sparse"] = sparse_zero

    adapted = {}
    for k in cfg.evaluation.few_shot_events:
        adapt_refs, eval_refs = few_shot_target_split(target_refs, k, cfg.train.seed + k)
        if not adapt_refs or not eval_refs:
            continue
        adapted[str(k)] = {}
        for frac in cfg.evaluation.sparse_sensor_fractions:
            reps = {}
            for seed in cfg.evaluation.sensor_layout_seeds:
                run_dir = out / f"adapt_k{k}_s{frac:g}_seed{seed}"
                ad_path = adapt_sparse_city(checkpoint, cfg, repo, target, adapt_refs, frac, run_dir,
                                            strategy="degree_stratified", epochs=adaptation_epochs, seed=seed)
                reps[str(seed)] = evaluate_sparse_adapted(checkpoint, ad_path, cfg, repo, eval_refs)
                atomic_json_dump({"adapt_events": [r.event_id for r in adapt_refs],
                                  "eval_events": [r.event_id for r in eval_refs]},
                                 run_dir / "target_partition.json")
            block = {"replicates": reps, "summary": summarize_replicates(reps)}
            dense_err = report["zero_shot_dense_warmup"].get("surface_depth_m_rmse", float("nan"))
            zsummary = sparse_zero[str(frac)]["summary"].get("surface_depth_m_rmse", {})
            asummary = block["summary"].get("surface_depth_m_rmse", {})
            if "mean" in zsummary and "mean" in asummary:
                block["transfer_recovery_surface_depth"] = transfer_recovery(zsummary["mean"], asummary["mean"], dense_err)
            adapted[str(k)][str(frac)] = block
    report["sparse_adaptation"] = adapted
    atomic_json_dump(report, out / "paper_protocol.json")
    return report
