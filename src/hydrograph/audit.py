from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .data import EventRef, UrbanFloodBenchRepository
from .schema import resolve_event_files, validate_split_schema
from .split import SplitManifest
from .utils import atomic_json_dump


@dataclass
class AuditFinding:
    severity: str
    code: str
    message: str


def audit_dataset(repo: UrbanFloodBenchRepository, split: str = "train", full: bool = True) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    try:
        refs = repo.list_events(split)
    except Exception as e:
        return [AuditFinding("P0", "SCHEMA_FATAL", str(e))]
    by_model: dict[str, list[EventRef]] = {}
    for r in refs:
        by_model.setdefault(r.model_id, []).append(r)
    for model, mrefs in by_model.items():
        split_dir = repo.root / model / split
        rep = validate_split_schema(split_dir, strict_event_edges=(split == "train"))
        for x in rep.missing_files:
            findings.append(AuditFinding("P0", "MISSING_FILE", f"{model}: {x}"))
        for x in rep.missing_columns:
            findings.append(AuditFinding("P0", "MISSING_COLUMN", f"{model}: {x}"))
        for x in rep.warnings:
            findings.append(AuditFinding("P2", "SCHEMA_WARNING", f"{model}: {x}"))
        try:
            graph = repo.load_static(model, split)
        except Exception as e:
            findings.append(AuditFinding("P0", "STATIC_LOAD", f"{model}: {e}"))
            continue
        refs_to_check = mrefs if full else mrefs[:3]
        for ref in refs_to_check:
            try:
                event = repo.load_event(ref, graph)
                if event.node1.shape[1] != graph.n1 or event.node2.shape[1] != graph.n2:
                    findings.append(AuditFinding("P0", "NODE_ALIGNMENT", f"{model}/{ref.event_id}"))
                if event.edge1.shape[1] != graph.edge1_index.shape[1] or event.edge2.shape[1] != graph.edge2_index.shape[1]:
                    findings.append(AuditFinding("P0", "EDGE_ALIGNMENT", f"{model}/{ref.event_id}"))
                for name, x in [("node1", event.node1), ("node2", event.node2), ("edge1", event.edge1), ("edge2", event.edge2)]:
                    if not torch.isfinite(x).all():
                        findings.append(AuditFinding("P0", "NONFINITE_DATA", f"{model}/{ref.event_id}/{name}"))
            except Exception as e:
                findings.append(AuditFinding("P0", "EVENT_LOAD", f"{model}/{ref.event_id}: {e}"))
    if not refs:
        findings.append(AuditFinding("P0", "NO_EVENTS", f"No {split} events found"))
    return findings


def audit_manifest(manifest: SplitManifest) -> list[AuditFinding]:
    try:
        manifest.validate_no_leakage()
        return []
    except Exception as e:
        return [AuditFinding("P0", "DATA_LEAKAGE", str(e))]


def audit_checkpoint(path: str | Path, manifest: SplitManifest | None = None) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        return [AuditFinding("P0", "CHECKPOINT_LOAD", str(e))]
    if ck.get("contract") != "HYDROGRAPH_CHECKPOINT_V1":
        findings.append(AuditFinding("P0", "CHECKPOINT_CONTRACT", "Unknown or missing checkpoint contract"))
    if "normalization" not in ck:
        findings.append(AuditFinding("P0", "NORMALIZATION_MISSING", "Checkpoint has no training normalization lineage"))
    if manifest and manifest.protocol == "leave_one_city_out" and manifest.target_model:
        fitted = set(ck.get("normalization", {}).get("fitted_models", []))
        if manifest.target_model in fitted:
            findings.append(AuditFinding("P0", "NORMALIZATION_LEAKAGE", "Held-out city appears in normalization statistics"))
    for key in ["model", "ema", "config_hash", "manifest", "feature_contract"]:
        if key not in ck:
            findings.append(AuditFinding("P1", "CHECKPOINT_LINEAGE", f"Missing checkpoint field: {key}"))
    return findings


def fail_on_severity(findings: list[AuditFinding], severities=("P0", "P1")) -> None:
    bad = [f for f in findings if f.severity in severities]
    if bad:
        raise RuntimeError("\n".join(f"[{f.severity}] {f.code}: {f.message}" for f in bad))


def write_audit(findings: list[AuditFinding], path: str | Path) -> None:
    atomic_json_dump([f.__dict__ for f in findings], path)
