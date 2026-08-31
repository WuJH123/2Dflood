from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .config import ExperimentConfig
from .data import EventData, EventRef, StaticGraph, UrbanFloodBenchRepository
from .losses import compute_step_loss
from .metrics import aggregate_event_balanced, event_metrics
from .model import HydroGraphOperator
from .normalization import NormalizationBundle, fit_normalization
from .physics import infer_dt_seconds
from .rollout import assimilate_sparse_history, inject_observations, rollout_event, warm_hidden
from .split import SplitManifest
from .utils import atomic_json_dump, choose_device, seed_everything, stable_hash


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for k, v in model.state_dict().items():
            if k not in self.shadow:
                self.shadow[k] = v.detach().clone()
            elif torch.is_floating_point(v):
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def state_dict(self):
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, state):
        self.shadow = {k: v.clone() for k, v in state.items()}

    def apply_to(self, model: torch.nn.Module):
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)
        return backup


def _feature_contract(graph: StaticGraph) -> dict:
    return {
        "node1_static": graph.node1_feature_names,
        "node2_static": graph.node2_feature_names,
        "edge1_static": graph.edge1_feature_names,
        "edge2_static": graph.edge2_feature_names,
        "node1_dynamic": ["water_level", "inlet_flow"],
        "node2_dynamic": ["rainfall", "water_level", "water_volume"],
        "edge_dynamic": ["flow", "velocity"],
    }


def build_model(cfg: ExperimentConfig, graph: StaticGraph, stats: NormalizationBundle | None = None) -> HydroGraphOperator:
    model = HydroGraphOperator(cfg.model, graph.node1_static.shape[-1], graph.node2_static.shape[-1],
                               graph.edge1_static.shape[-1], graph.edge2_static.shape[-1])
    if stats is not None:
        model.set_normalization(stats)
    return model


def teacher_forcing_ratio(epoch: int, cfg: ExperimentConfig) -> float:
    tr = cfg.train
    p = min(1.0, max(0.0, epoch / max(1, tr.curriculum_epochs)))
    return tr.teacher_forcing_start + p * (tr.teacher_forcing_end - tr.teacher_forcing_start)


def _window_starts(event: EventData, cfg: ExperimentConfig, rng: random.Random) -> list[int]:
    c, h, stride = cfg.data.context_steps, cfg.data.horizon_steps, cfg.data.window_stride
    candidates = list(range(max(0, c - 1), event.timesteps - h - 1, max(1, stride)))
    if not candidates:
        return []
    cap = cfg.train.windows_per_event
    if cap > 0 and len(candidates) > cap:
        candidates = rng.sample(candidates, cap)
    return sorted(candidates)


def _sensor_masks(graph: StaticGraph, cfg: ExperimentConfig, rng: random.Random, device: torch.device):
    if rng.random() >= cfg.train.masked_state_prob:
        return (torch.ones(graph.n1, dtype=torch.bool, device=device),
                torch.ones(graph.n2, dtype=torch.bool, device=device))
    keep = max(0.001, 1.0 - cfg.train.masked_sensor_prob)
    m1 = torch.rand(graph.n1, device=device) < keep
    m2 = torch.rand(graph.n2, device=device) < keep
    if not m1.any():
        m1[rng.randrange(graph.n1)] = True
    if not m2.any():
        m2[rng.randrange(graph.n2)] = True
    return m1, m2


def train_window(
    model: HydroGraphOperator,
    graph: StaticGraph,
    event: EventData,
    start: int,
    cfg: ExperimentConfig,
    tf_ratio: float,
    rng: random.Random,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = event.node1.device
    mask1, mask2 = _sensor_masks(graph, cfg, rng, device)
    sparse = not bool(mask1.all() and mask2.all())
    if sparse:
        (cur1, cur2, cure1, cure2), hidden = assimilate_sparse_history(
            model, graph, event, start, cfg.data.context_steps, mask1, mask2
        )
    else:
        hidden = None
        if start > 0 and cfg.data.context_steps > 1:
            hidden = warm_hidden(model, graph, event, start - 1, cfg.data.context_steps - 1)
        cur1, cur2, cure1, cure2 = event.node1[start], event.node2[start], event.edge1[start], event.edge2[start]
    dt = infer_dt_seconds(event)
    total = torch.zeros((), device=device)
    term_sums: dict[str, float] = {}
    used = 0
    for step in range(cfg.data.horizon_steps):
        target_t = start + step + 1
        if target_t >= event.timesteps:
            break
        pred = model.forward_step(graph, cur1, cur2, cure1, cure2, event.node2[target_t, :, 0],
                                  mask1, mask2, hidden)
        loss = compute_step_loss(pred, event.node1[target_t], event.node2[target_t], event.edge1[target_t],
                                 event.edge2[target_t], cur2, graph, cfg.loss, dt)
        total = total + loss.total
        for k, v in loss.terms.items():
            term_sums[k] = term_sums.get(k, 0.0) + float(v.detach().cpu())
        used += 1
        use_truth = rng.random() < tf_ratio
        if use_truth:
            if sparse:
                cur1, cur2, cure1, cure2 = inject_observations(
                    (pred.node1, pred.node2, pred.edge1, pred.edge2), event, target_t,
                    mask1, mask2, observe_edges=False
                )
            else:
                cur1, cur2, cure1, cure2 = (event.node1[target_t], event.node2[target_t],
                                            event.edge1[target_t], event.edge2[target_t])
        else:
            cur1, cur2, cure1, cure2 = pred.node1, pred.node2, pred.edge1, pred.edge2
        hidden = (pred.hidden1, pred.hidden2)
    if used == 0:
        raise ValueError("Training window has zero forecast steps")
    return total / used, {k: v / used for k, v in term_sums.items()}


@torch.no_grad()
def validate(
    model: HydroGraphOperator,
    repo: UrbanFloodBenchRepository,
    refs: list[EventRef],
    cfg: ExperimentConfig,
    device: torch.device,
    limit: int = 0,
) -> tuple[float, dict[str, float]]:
    if not refs:
        return float("nan"), {}
    rows = []
    graphs: dict[str, StaticGraph] = {}
    eval_refs = refs[:limit] if limit > 0 else refs
    for ref in eval_refs:
        g = graphs.setdefault(ref.model_id, repo.load_static(ref.model_id, ref.split).to(device))
        e = repo.load_event(ref).to(device)
        result = rollout_event(model, g, e, cfg.data.warmup_steps, cfg.data.context_steps)
        rows.append(event_metrics(g, e, result, cfg.evaluation.flood_thresholds_m, cfg.evaluation.arrival_threshold_m))
    agg = aggregate_event_balanced(rows)
    objective = 0.5 * (agg.get("water_level_1d_rmse", math.inf) + agg.get("water_level_2d_rmse", math.inf))
    return objective, agg


def fit_source_normalization(repo: UrbanFloodBenchRepository, refs: list[EventRef]) -> NormalizationBundle:
    return fit_normalization(repo.iter_events(refs))


def train(
    cfg: ExperimentConfig,
    repo: UrbanFloodBenchRepository,
    manifest: SplitManifest,
    resume: str | None = None,
) -> Path:
    seed_everything(cfg.train.seed)
    device = choose_device(cfg.train.device)
    out = Path(cfg.train.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest.validate_no_leakage()
    atomic_json_dump(manifest.serializable(), out / "split_manifest.json")
    atomic_json_dump(cfg.to_dict(), out / "resolved_config.json")

    first_graph = repo.load_static(manifest.train[0].model_id, manifest.train[0].split)
    stats = fit_source_normalization(repo, manifest.train)
    # Fail closed if held-out target city leaked into normalization.
    if manifest.protocol == "leave_one_city_out" and manifest.target_model in stats.fitted_models:
        raise RuntimeError("P0 leakage: target city appears in source normalization statistics")
    model = build_model(cfg, first_graph, stats).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.train.amp and device.type == "cuda")
    ema = EMA(model, cfg.train.ema_decay)
    start_epoch, best, bad_epochs = 0, math.inf, 0

    if resume:
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        if ck["config_hash"] != stable_hash(cfg.to_dict()):
            raise ValueError("Resume config hash mismatch; refuse to silently change experiment contract")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        ema.load_state_dict(ck["ema"])
        start_epoch = ck["epoch"] + 1
        best = ck["best"]
        bad_epochs = ck.get("bad_epochs", 0)

    log_path = out / "train.jsonl"
    rng = random.Random(cfg.train.seed)
    graph_cache: dict[str, StaticGraph] = {}
    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        refs = manifest.train[:]
        rng.shuffle(refs)
        tf = teacher_forcing_ratio(epoch, cfg)
        epoch_losses: list[float] = []
        t0 = time.time()
        for ref in refs:
            graph = graph_cache.setdefault(ref.model_id, repo.load_static(ref.model_id, ref.split).to(device))
            event = repo.load_event(ref).to(device)
            starts = _window_starts(event, cfg, rng)
            for start in starts:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=cfg.train.amp and device.type == "cuda"):
                    loss, _ = train_window(model, graph, event, start, cfg, tf, rng)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at {ref.model_id}/{ref.event_id} t={start}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                ema.update(model)
                epoch_losses.append(float(loss.detach().cpu()))
            del event

        backup = ema.apply_to(model)
        val_refs = manifest.val[:]
        rng.shuffle(val_refs)
        val_obj, val_metrics = validate(model, repo, val_refs, cfg, device, cfg.train.val_events_per_epoch)
        model.load_state_dict(backup)
        train_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        improved = val_obj < best if math.isfinite(val_obj) else train_loss < best
        score = val_obj if math.isfinite(val_obj) else train_loss
        if improved:
            best, bad_epochs = score, 0
        else:
            bad_epochs += 1
        record = {"epoch": epoch, "train_loss": train_loss, "val_objective": val_obj,
                  "teacher_forcing": tf, "best": best, "seconds": time.time() - t0, **val_metrics}
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

        checkpoint = {
            "contract": "HYDROGRAPH_CHECKPOINT_V1",
            "epoch": epoch, "best": best, "bad_epochs": bad_epochs,
            "model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "normalization": stats.state_dict(),
            "config": cfg.to_dict(), "config_hash": stable_hash(cfg.to_dict()),
            "manifest": manifest.serializable(), "feature_contract": _feature_contract(first_graph),
        }
        torch.save(checkpoint, out / "last.pt")
        if improved:
            torch.save(checkpoint, out / "best.pt")
        if (epoch + 1) % cfg.train.checkpoint_every == 0:
            torch.save(checkpoint, out / f"epoch_{epoch+1:04d}.pt")
        if bad_epochs >= cfg.train.early_stopping_patience:
            break
    return out / "best.pt"


def load_model_checkpoint(path: str | Path, device: torch.device, graph: StaticGraph) -> tuple[HydroGraphOperator, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("contract") != "HYDROGRAPH_CHECKPOINT_V1":
        raise ValueError("Unsupported checkpoint contract")
    cfg_dict = ck["config"]
    # Avoid YAML round-trip here; reconstruct via dataclass constructors.
    from .config import DataConfig, EvaluationConfig, LossConfig, ModelConfig, TrainConfig
    cfg = ExperimentConfig(
        data=DataConfig(**cfg_dict["data"]), model=ModelConfig(**cfg_dict["model"]),
        loss=LossConfig(**cfg_dict["loss"]), train=TrainConfig(**cfg_dict["train"]),
        evaluation=EvaluationConfig(**cfg_dict["evaluation"]), protocol=cfg_dict["protocol"],
        target_model=cfg_dict.get("target_model"),
    )
    stats = NormalizationBundle.from_state_dict(ck["normalization"])
    model = build_model(cfg, graph, stats)
    state = ck.get("ema", ck["model"])
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, ck
