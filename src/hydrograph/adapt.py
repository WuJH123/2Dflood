from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from .config import ExperimentConfig
from .data import EventRef, StaticGraph, UrbanFloodBenchRepository
from .metrics import aggregate_event_balanced, event_metrics
from .physics import infer_dt_seconds, surface_mass_residual
from .rollout import assimilate_sparse_history, inject_observations, rollout_event
from .trainer import load_model_checkpoint
from .utils import atomic_json_dump, choose_device, seed_everything


def select_sensor_masks(graph: StaticGraph, fraction: float, seed: int, strategy: str = "random") -> tuple[torch.Tensor, torch.Tensor]:
    if not 0 < fraction <= 1:
        raise ValueError("sensor fraction must be in (0, 1]")
    rng = random.Random(seed)

    def pick(n: int, degree: torch.Tensor | None = None):
        k = max(1, int(round(n * fraction)))
        if strategy == "random" or degree is None:
            ids = rng.sample(range(n), min(k, n))
        elif strategy == "degree_stratified":
            # Split by degree quartiles and sample proportionally, preserving broad hydraulic coverage.
            order = torch.argsort(degree).tolist()
            bins = [order[i::4] for i in range(4)]
            ids = []
            for b in bins:
                if not b:
                    continue
                kb = max(1, int(round(k * len(b) / n)))
                ids.extend(rng.sample(b, min(kb, len(b))))
            ids = list(dict.fromkeys(ids))[:k]
            if len(ids) < k:
                pool = [i for i in range(n) if i not in set(ids)]
                ids.extend(rng.sample(pool, min(k - len(ids), len(pool))))
        else:
            raise ValueError(f"Unknown sensor strategy: {strategy}")
        mask = torch.zeros(n, dtype=torch.bool)
        mask[ids] = True
        return mask

    d1 = torch.bincount(graph.edge1_index.flatten(), minlength=graph.n1)
    d2 = torch.bincount(graph.edge2_index.flatten(), minlength=graph.n2)
    return pick(graph.n1, d1), pick(graph.n2, d2)


def _sparse_adaptation_loss(pred, event, t: int, graph: StaticGraph, mask1: torch.Tensor,
                            mask2: torch.Tensor, current2: torch.Tensor, dt: float) -> torch.Tensor:
    # Target-city adaptation sees water-level sensors only. It never uses full-field labels,
    # target edge truth, target volume truth, or future states at unobserved nodes.
    l1 = torch.sqrt(torch.mean((pred.node1[mask1, 0] - event.node1[t, mask1, 0]).square()) + 1e-8)
    l2 = torch.sqrt(torch.mean((pred.node2[mask2, 1] - event.node2[t, mask2, 1]).square()) + 1e-8)
    mass = surface_mass_residual(graph, current2, pred.node1, pred.node2, pred.edge2, dt)
    return 0.5 * (l1 + l2) + 0.10 * mass.local_relative + 0.05 * mass.global_relative


def adapt_sparse_city(
    checkpoint: str | Path,
    cfg: ExperimentConfig,
    repo: UrbanFloodBenchRepository,
    target_model: str,
    adaptation_refs: list[EventRef],
    sensor_fraction: float,
    output_dir: str | Path,
    strategy: str = "random",
    epochs: int = 20,
    seed: int | None = None,
) -> Path:
    if not adaptation_refs:
        raise ValueError("Sparse adaptation requires at least one target-city adaptation event")
    if any(r.model_id != target_model for r in adaptation_refs):
        raise ValueError("Adaptation refs must all belong to target_model")
    seed = cfg.train.seed if seed is None else seed
    seed_everything(seed)
    device = choose_device(cfg.train.device)
    graph_cpu = repo.load_static(target_model, adaptation_refs[0].split)
    graph = graph_cpu.to(device)
    model, ck = load_model_checkpoint(checkpoint, device, graph_cpu)
    source_models = set(ck["normalization"]["fitted_models"])
    if target_model in source_models and ck["manifest"]["protocol"] == "leave_one_city_out":
        raise RuntimeError("P0 leakage: target city already present in source normalization/checkpoint")
    model.freeze_for_adaptation(train_heads=True)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable adapter/head parameters after freezing backbone")
    optimizer = torch.optim.AdamW(params, lr=cfg.train.lr * 0.5, weight_decay=cfg.train.weight_decay)
    mask1_cpu, mask2_cpu = select_sensor_masks(graph_cpu, sensor_fraction, seed, strategy)
    mask1, mask2 = mask1_cpu.to(device), mask2_cpu.to(device)
    rng = random.Random(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json_dump({
        "contract": "SPARSE_CITY_ADAPT_V1", "target_model": target_model,
        "sensor_fraction": sensor_fraction, "sensor_strategy": strategy,
        "node1_sensor_ids": graph_cpu.node1_ids[mask1_cpu].tolist(),
        "node2_sensor_ids": graph_cpu.node2_ids[mask2_cpu].tolist(),
        "adaptation_events": [r.event_id for r in adaptation_refs],
        "source_checkpoint": str(checkpoint), "seed": seed,
    }, out / "adaptation_manifest.json")

    log = out / "adapt.jsonl"
    for epoch in range(epochs):
        model.train()
        refs = adaptation_refs[:]
        rng.shuffle(refs)
        losses = []
        for ref in refs:
            event = repo.load_event(ref).to(device)
            dt = infer_dt_seconds(event)
            # A historical sensor record is assimilated causally; only sensor values are inserted.
            starts = list(range(max(1, cfg.data.context_steps - 1), event.timesteps - cfg.data.horizon_steps - 1,
                                max(1, cfg.data.window_stride)))
            if cfg.train.windows_per_event > 0 and len(starts) > cfg.train.windows_per_event:
                starts = rng.sample(starts, cfg.train.windows_per_event)
            for start in starts:
                optimizer.zero_grad(set_to_none=True)
                (cur1, cur2, cure1, cure2), hidden = assimilate_sparse_history(
                    model, graph, event, start, cfg.data.context_steps, mask1, mask2
                )
                loss = torch.zeros((), device=device)
                used = 0
                for step in range(cfg.data.horizon_steps):
                    t = start + step + 1
                    if t >= event.timesteps:
                        break
                    pred = model.forward_step(graph, cur1, cur2, cure1, cure2, event.node2[t, :, 0],
                                              mask1, mask2, hidden)
                    loss = loss + _sparse_adaptation_loss(pred, event, t, graph, mask1, mask2, cur2, dt)
                    used += 1
                    cur1, cur2, cure1, cure2 = inject_observations(
                        (pred.node1, pred.node2, pred.edge1, pred.edge2), event, t,
                        mask1, mask2, observe_edges=False
                    )
                    hidden = (pred.hidden1, pred.hidden2)
                if used:
                    loss = loss / used
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite sparse adaptation loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, "loss": sum(losses) / max(1, len(losses))}) + "\n")

    payload = {
        "contract": "HYDROGRAPH_ADAPTER_V1",
        "base_checkpoint": str(checkpoint),
        "target_model": target_model,
        "sensor_fraction": sensor_fraction,
        "sensor_strategy": strategy,
        "sensor_mask1": mask1_cpu,
        "sensor_mask2": mask2_cpu,
        "model": model.state_dict(),
        "adaptation_events": [r.event_id for r in adaptation_refs],
    }
    path = out / "adapted.pt"
    torch.save(payload, path)
    return path


@torch.no_grad()
def evaluate_sparse_adapted(
    base_checkpoint: str | Path,
    adapted_checkpoint: str | Path,
    cfg: ExperimentConfig,
    repo: UrbanFloodBenchRepository,
    refs: list[EventRef],
) -> dict[str, float]:
    if not refs:
        return {}
    device = choose_device(cfg.train.device)
    g0 = repo.load_static(refs[0].model_id, refs[0].split)
    graph = g0.to(device)
    model, _ = load_model_checkpoint(base_checkpoint, device, g0)
    ad = torch.load(adapted_checkpoint, map_location="cpu", weights_only=False)
    if ad.get("contract") != "HYDROGRAPH_ADAPTER_V1":
        raise ValueError("Unsupported adapter checkpoint")
    model.load_state_dict(ad["model"], strict=True)
    mask1, mask2 = ad["sensor_mask1"].to(device), ad["sensor_mask2"].to(device)
    rows = []
    for ref in refs:
        if ref.model_id != g0.model_id:
            raise ValueError("Sparse adapted evaluation refs must be from one target city")
        event = repo.load_event(ref).to(device)
        result = rollout_event(model, graph, event, cfg.data.warmup_steps, cfg.data.context_steps, mask1, mask2)
        rows.append(event_metrics(graph, event, result, cfg.evaluation.flood_thresholds_m,
                                  cfg.evaluation.arrival_threshold_m))
    return aggregate_event_balanced(rows)
