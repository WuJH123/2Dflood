from __future__ import annotations

import argparse
import copy
from pathlib import Path

from hydrograph.config import load_config
from hydrograph.data import UrbanFloodBenchRepository
from hydrograph.paper_eval import run_target_protocol
from hydrograph.split import make_manifest
from hydrograph.trainer import train


def main():
    ap = argparse.ArgumentParser(description="Run leave-one-city-out training + target-city evidence protocol.")
    ap.add_argument("--config", default="configs/paper.yaml")
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--adapt-epochs", type=int, default=20)
    args = ap.parse_args()
    base_cfg = load_config(args.config)
    repo = UrbanFloodBenchRepository(base_cfg.data.root, base_cfg.data.model_glob, base_cfg.data.strict_schema)
    refs = repo.list_events("train")
    models = sorted({r.model_id for r in refs})
    targets = args.targets or models
    for target in targets:
        cfg = copy.deepcopy(base_cfg)
        cfg.protocol = "leave_one_city_out"
        cfg.target_model = target
        cfg.train.output_dir = str(Path(base_cfg.train.output_dir) / f"loco_{target}")
        manifest = make_manifest(refs, cfg.protocol, cfg.data.split_seed, target_model=target)
        checkpoint = train(cfg, repo, manifest)
        run_target_protocol(checkpoint, cfg, repo, manifest.test,
                            Path(cfg.train.output_dir) / "paper_evidence", args.adapt_epochs)


if __name__ == "__main__":
    main()
