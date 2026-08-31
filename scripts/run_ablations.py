from __future__ import annotations

import argparse
import copy
from pathlib import Path

from hydrograph.config import load_config
from hydrograph.data import UrbanFloodBenchRepository
from hydrograph.paper_eval import evaluate_checkpoint
from hydrograph.split import make_manifest
from hydrograph.trainer import train
from hydrograph.utils import atomic_json_dump


def apply_ablation(cfg, name: str):
    if name == "full":
        return
    if name == "no_global_tokens":
        cfg.model.use_global_tokens = False
    elif name == "no_coupling":
        cfg.model.use_coupling = False
    elif name == "no_hydraulic_head":
        cfg.model.use_hydraulic_head = False
    elif name == "no_flux_supervision":
        cfg.model.use_flux_decoder = False
        cfg.loss.edge_flow_1d = cfg.loss.edge_flow_2d = 0.0
        cfg.loss.edge_velocity_1d = cfg.loss.edge_velocity_2d = 0.0
        cfg.loss.mass_local = cfg.loss.mass_global = 0.0
    elif name == "no_mass_loss":
        cfg.loss.mass_local = cfg.loss.mass_global = 0.0
    elif name == "no_masked_pretraining":
        cfg.train.masked_state_prob = 0.0
    else:
        raise ValueError(f"Unknown ablation: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paper.yaml")
    ap.add_argument("--target-model", required=True)
    ap.add_argument("--names", nargs="+", default=["full", "no_global_tokens", "no_coupling", "no_hydraulic_head", "no_flux_supervision", "no_mass_loss", "no_masked_pretraining"])
    ap.add_argument("--output", default="outputs/ablations")
    args = ap.parse_args()
    base = load_config(args.config)
    repo = UrbanFloodBenchRepository(base.data.root, base.data.model_glob, base.data.strict_schema)
    refs = repo.list_events("train")
    results = {}
    for name in args.names:
        cfg = copy.deepcopy(base)
        cfg.protocol = "leave_one_city_out"
        cfg.target_model = args.target_model
        apply_ablation(cfg, name)
        cfg.train.output_dir = str(Path(args.output) / args.target_model / name)
        manifest = make_manifest(refs, cfg.protocol, cfg.data.split_seed, cfg.target_model)
        ck = train(cfg, repo, manifest)
        results[name] = evaluate_checkpoint(ck, cfg, repo, manifest.test, Path(cfg.train.output_dir) / "eval")
    atomic_json_dump(results, Path(args.output) / args.target_model / "ablation_summary.json")


if __name__ == "__main__":
    main()
