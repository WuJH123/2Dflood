from __future__ import annotations

import argparse
import json
import time

import torch

from hydrograph.config import load_config
from hydrograph.data import UrbanFloodBenchRepository
from hydrograph.rollout import rollout_event
from hydrograph.trainer import load_model_checkpoint
from hydrograph.utils import choose_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paper.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--event-id")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    cfg = load_config(args.config)
    repo = UrbanFloodBenchRepository(cfg.data.root, cfg.data.model_glob, cfg.data.strict_schema)
    refs = [r for r in repo.list_events("train") if r.model_id == args.model_id]
    if args.event_id:
        refs = [r for r in refs if r.event_id == args.event_id]
    ref = refs[0]
    device = choose_device(cfg.train.device)
    g0 = repo.load_static(ref.model_id, ref.split)
    g = g0.to(device)
    e = repo.load_event(ref).to(device)
    model, _ = load_model_checkpoint(args.checkpoint, device, g0)
    times = []
    for _ in range(args.repeats + 1):
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        rollout_event(model, g, e, cfg.data.warmup_steps, cfg.data.context_steps)
        if device.type == "cuda": torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times = times[1:]
    forecast_steps = e.timesteps - cfg.data.warmup_steps
    print(json.dumps({
        "device": str(device), "event": f"{ref.model_id}/{ref.event_id}",
        "forecast_steps": forecast_steps, "seconds_mean": sum(times) / len(times),
        "seconds_per_step": sum(times) / len(times) / forecast_steps,
        "speed_hz": forecast_steps / (sum(times) / len(times)),
    }, indent=2))


if __name__ == "__main__":
    main()
