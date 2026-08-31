from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapt import adapt_sparse_city, evaluate_sparse_adapted
from .audit import audit_dataset, fail_on_severity, write_audit
from .config import load_config
from .data import UrbanFloodBenchRepository
from .paper_eval import evaluate_checkpoint, run_target_protocol
from .schema import validate_split_schema
from .split import few_shot_target_split, make_manifest
from .trainer import train


def _repo(cfg):
    return UrbanFloodBenchRepository(cfg.data.root, cfg.data.model_glob, cfg.data.strict_schema)


def cmd_inspect(args):
    cfg = load_config(args.config)
    repo = _repo(cfg)
    rows = []
    for model in repo.model_dirs():
        split_dir = model / args.split
        report = validate_split_schema(split_dir, strict_event_edges=(args.split == "train"))
        refs = [r for r in repo.list_events(args.split) if r.model_id == model.name]
        graph = repo.load_static(model.name, args.split) if split_dir.exists() else None
        rows.append({
            "model": model.name, "events": len(refs), "schema_ok": report.ok,
            "warnings": list(report.warnings),
            "n1": graph.n1 if graph else None, "n2": graph.n2 if graph else None,
            "e1": graph.edge1_index.shape[1] if graph else None,
            "e2": graph.edge2_index.shape[1] if graph else None,
            "couplings": graph.coupling_index.shape[1] if graph else None,
            "features": {
                "node1": graph.node1_feature_names if graph else [], "node2": graph.node2_feature_names if graph else [],
                "edge1": graph.edge1_feature_names if graph else [], "edge2": graph.edge2_feature_names if graph else [],
            },
        })
    print(json.dumps(rows, indent=2))


def cmd_audit(args):
    cfg = load_config(args.config)
    repo = _repo(cfg)
    findings = audit_dataset(repo, args.split, full=args.full)
    if args.output:
        write_audit(findings, args.output)
    for f in findings:
        print(f"[{f.severity}] {f.code}: {f.message}")
    if not findings:
        print("PASS: no dataset audit findings")
    if args.fail_on_p1:
        fail_on_severity(findings, ("P0", "P1"))
    else:
        fail_on_severity(findings, ("P0",))


def cmd_train(args):
    cfg = load_config(args.config)
    if args.target_model:
        cfg.target_model = args.target_model
        cfg.protocol = "leave_one_city_out"
    if args.output_dir:
        cfg.train.output_dir = args.output_dir
    repo = _repo(cfg)
    refs = repo.list_events("train")
    manifest = make_manifest(refs, cfg.protocol, cfg.data.split_seed, cfg.target_model)
    path = train(cfg, repo, manifest, resume=args.resume)
    print(path)


def cmd_evaluate(args):
    cfg = load_config(args.config)
    repo = _repo(cfg)
    refs = repo.list_events(args.split)
    if args.target_model:
        refs = [r for r in refs if r.model_id == args.target_model]
    metrics = evaluate_checkpoint(args.checkpoint, cfg, repo, refs, args.output_dir,
                                  sparse_fraction=args.sparse_fraction,
                                  sensor_seed=args.sensor_seed)
    print(json.dumps(metrics, indent=2, sort_keys=True))


def cmd_adapt(args):
    cfg = load_config(args.config)
    repo = _repo(cfg)
    target_refs = [r for r in repo.list_events("train") if r.model_id == args.target_model]
    adapt_refs, eval_refs = few_shot_target_split(target_refs, args.k_events, cfg.train.seed + args.k_events)
    path = adapt_sparse_city(args.checkpoint, cfg, repo, args.target_model, adapt_refs,
                             args.sensor_fraction, args.output_dir,
                             strategy=args.sensor_strategy, epochs=args.epochs)
    metrics = evaluate_sparse_adapted(args.checkpoint, path, cfg, repo, eval_refs)
    print(json.dumps({"adapter": str(path), "adapt_events": [r.event_id for r in adapt_refs],
                      "eval_events": [r.event_id for r in eval_refs], "metrics": metrics}, indent=2))


def cmd_paper(args):
    cfg = load_config(args.config)
    repo = _repo(cfg)
    refs = [r for r in repo.list_events("train") if r.model_id == args.target_model]
    report = run_target_protocol(args.checkpoint, cfg, repo, refs, args.output_dir, args.adapt_epochs)
    print(json.dumps(report, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hydrograph", description="HydroGraph-Operator research pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="configs/paper.yaml")

    x = sub.add_parser("inspect", parents=[common])
    x.add_argument("--split", default="train")
    x.set_defaults(func=cmd_inspect)

    x = sub.add_parser("audit", parents=[common])
    x.add_argument("--split", default="train")
    x.add_argument("--full", action="store_true")
    x.add_argument("--output")
    x.add_argument("--fail-on-p1", action="store_true")
    x.set_defaults(func=cmd_audit)

    x = sub.add_parser("train", parents=[common])
    x.add_argument("--target-model")
    x.add_argument("--output-dir")
    x.add_argument("--resume")
    x.set_defaults(func=cmd_train)

    x = sub.add_parser("evaluate", parents=[common])
    x.add_argument("--checkpoint", required=True)
    x.add_argument("--split", default="train")
    x.add_argument("--target-model")
    x.add_argument("--output-dir")
    x.add_argument("--sparse-fraction", type=float)
    x.add_argument("--sensor-seed", type=int)
    x.set_defaults(func=cmd_evaluate)

    x = sub.add_parser("adapt", parents=[common])
    x.add_argument("--checkpoint", required=True)
    x.add_argument("--target-model", required=True)
    x.add_argument("--k-events", type=int, default=3)
    x.add_argument("--sensor-fraction", type=float, default=0.01)
    x.add_argument("--sensor-strategy", choices=["random", "degree_stratified"], default="degree_stratified")
    x.add_argument("--epochs", type=int, default=20)
    x.add_argument("--output-dir", required=True)
    x.set_defaults(func=cmd_adapt)

    x = sub.add_parser("paper-protocol", parents=[common])
    x.add_argument("--checkpoint", required=True)
    x.add_argument("--target-model", required=True)
    x.add_argument("--output-dir", required=True)
    x.add_argument("--adapt-epochs", type=int, default=20)
    x.set_defaults(func=cmd_paper)
    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
