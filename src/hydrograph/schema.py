from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


STATIC_1D_NODE_REQUIRED = [
    "node_idx", "depth", "invert_elevation", "surface_elevation", "base_area"
]
STATIC_2D_NODE_REQUIRED = [
    "node_idx", "area", "roughness", "min_elevation", "elevation", "aspect", "curvature"
]
STATIC_1D_EDGE_REQUIRED = ["edge_idx", "relative_position_x", "relative_position_y", "length", "diameter", "shape", "roughness", "slope"]
STATIC_2D_EDGE_REQUIRED = ["edge_idx", "relative_position_x", "relative_position_y", "face_length", "length", "slope"]
EDGE_INDEX_REQUIRED = ["from_node", "to_node"]
COUPLING_REQUIRED = ["node_1d", "node_2d"]
DYNAMIC_1D_NODE_REQUIRED = ["timestep", "node_idx", "water_level", "inlet_flow"]
DYNAMIC_2D_NODE_REQUIRED = ["timestep", "node_idx", "rainfall", "water_level", "water_volume"]
DYNAMIC_EDGE_REQUIRED = ["timestep", "edge_idx", "flow", "velocity"]
TIMESTEP_MIN_COLUMNS = ["timestep"]


@dataclass(frozen=True)
class SchemaReport:
    ok: bool
    missing_files: tuple[str, ...]
    missing_columns: tuple[str, ...]
    warnings: tuple[str, ...]

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise ValueError(
                "UrbanFloodBench schema validation failed:\n"
                + "\n".join([*self.missing_files, *self.missing_columns])
            )


def _resolve_file(folder: Path, candidates: Iterable[str]) -> Path | None:
    by_norm = {p.name.lower().replace(" ", ""): p for p in folder.glob("*.csv")}
    for name in candidates:
        hit = by_norm.get(name.lower().replace(" ", ""))
        if hit is not None:
            return hit
    return None


def resolve_static_files(split_dir: str | Path) -> dict[str, Path]:
    split_dir = Path(split_dir)
    specs = {
        "nodes1": ["1d_nodes_static.csv"],
        "nodes2": ["2d_nodes_static.csv", "2d_nodes_index.csv"],
        "edges1": ["1d_edges_static.csv"],
        "edges2": ["2d_edges_static.csv", "2d_edges_index.csv"],
        "index1": ["1d_edge_index.csv"],
        "index2": ["2d_edge_index.csv"],
        "coupling": ["1d2d_connections.csv", "1d_2d_connections.csv"],
    }
    out: dict[str, Path] = {}
    for key, candidates in specs.items():
        p = _resolve_file(split_dir, candidates)
        if p is not None:
            out[key] = p
    return out


def resolve_event_files(event_dir: str | Path) -> dict[str, Path]:
    event_dir = Path(event_dir)
    specs = {
        "nodes1": ["1d_nodes_dynamic_all.csv"],
        "nodes2": ["2d_nodes_dynamic_all.csv"],
        "edges1": ["1d_edges_dynamic_all.csv"],
        "edges2": ["2d_edges_dynamic_all.csv"],
        "timesteps": ["timesteps.csv"],
    }
    out: dict[str, Path] = {}
    for key, candidates in specs.items():
        p = _resolve_file(event_dir, candidates)
        if p is not None:
            out[key] = p
    return out


def _check_cols(path: Path, required: list[str], label: str, missing: list[str], warnings: list[str]) -> None:
    df = pd.read_csv(path, nrows=8)
    cols = set(df.columns)
    aliases = {"2d_flow": "flow", "2d_velocity": "velocity", "1d_edge_flow": "flow", "1d_edge_velocity": "velocity"}
    cols = cols | {aliases[c] for c in cols if c in aliases}
    absent = [c for c in required if c not in cols]
    if absent:
        missing.append(f"{label}: missing columns {absent} in {path.name}")
    if df.columns.duplicated().any():
        warnings.append(f"{label}: duplicate column names in {path.name}")


def validate_split_schema(split_dir: str | Path, strict_event_edges: bool = True) -> SchemaReport:
    split_dir = Path(split_dir)
    missing_files: list[str] = []
    missing_columns: list[str] = []
    warnings: list[str] = []
    static = resolve_static_files(split_dir)
    for k in ["nodes1", "nodes2", "index1", "index2", "coupling"]:
        if k not in static:
            missing_files.append(f"missing static file group: {k}")
    if "nodes1" in static:
        _check_cols(static["nodes1"], STATIC_1D_NODE_REQUIRED, "1D nodes static", missing_columns, warnings)
    if "nodes2" in static:
        # Some released variants omit optional topographic fields; core loader handles this fail-soft.
        _check_cols(static["nodes2"], ["node_idx"], "2D nodes static", missing_columns, warnings)
    if "index1" in static:
        _check_cols(static["index1"], EDGE_INDEX_REQUIRED, "1D edge index", missing_columns, warnings)
    if "index2" in static:
        _check_cols(static["index2"], EDGE_INDEX_REQUIRED, "2D edge index", missing_columns, warnings)
    if "coupling" in static:
        _check_cols(static["coupling"], COUPLING_REQUIRED, "1D-2D coupling", missing_columns, warnings)

    events = sorted(p for p in split_dir.iterdir() if p.is_dir() and p.name.lower().startswith("event")) if split_dir.exists() else []
    if not events:
        warnings.append("no event_* directories found")
    for event in events[:3]:  # sample first three for structural audit; full audit is in audit.py
        ef = resolve_event_files(event)
        for k in ["nodes1", "nodes2", "timesteps"]:
            if k not in ef:
                missing_files.append(f"{event.name}: missing event file group {k}")
        if strict_event_edges:
            for k in ["edges1", "edges2"]:
                if k not in ef:
                    missing_files.append(f"{event.name}: missing event file group {k}")
        if "nodes1" in ef:
            _check_cols(ef["nodes1"], DYNAMIC_1D_NODE_REQUIRED, f"{event.name}/1D dynamic", missing_columns, warnings)
        if "nodes2" in ef:
            _check_cols(ef["nodes2"], DYNAMIC_2D_NODE_REQUIRED, f"{event.name}/2D dynamic", missing_columns, warnings)
        for k in ["edges1", "edges2"]:
            if k in ef:
                _check_cols(ef[k], ["timestep", "edge_idx"], f"{event.name}/{k}", missing_columns, warnings)
    return SchemaReport(
        ok=not missing_files and not missing_columns,
        missing_files=tuple(missing_files),
        missing_columns=tuple(missing_columns),
        warnings=tuple(warnings),
    )
