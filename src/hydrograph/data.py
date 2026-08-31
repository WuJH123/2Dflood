from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch

from .schema import resolve_event_files, resolve_static_files, validate_split_schema


NODE1_STATIC_SPEC = [
    ("depth", ["depth"]), ("invert_elevation", ["invert_elevation"]),
    ("surface_elevation", ["surface_elevation"]), ("base_area", ["base_area"]),
]
NODE2_STATIC_SPEC = [
    ("position_x", ["2d_position_x", "position_x", "x"]),
    ("position_y", ["2d_position_y", "position_y", "y"]),
    ("area", ["area"]), ("roughness", ["roughness"]),
    ("min_elevation", ["min_elevation"]),
    ("elevation", ["elevation", "centroid_elevation"]),
    ("aspect", ["aspect"]), ("curvature", ["curvature"]),
    ("flow_accumulation", ["flow_accumulation"]),
]
EDGE1_STATIC_SPEC = [
    ("relative_position_x", ["relative_position_x"]),
    ("relative_position_y", ["relative_position_y"]),
    ("length", ["length"]), ("diameter", ["diameter"]),
    ("roughness", ["roughness"]), ("slope", ["slope"]),
]
EDGE2_STATIC_SPEC = [
    ("relative_position_x", ["relative_position_x"]),
    ("relative_position_y", ["relative_position_y"]),
    ("face_length", ["face_length"]),
    ("length", ["length", "2d_length"]), ("slope", ["slope"]),
]


def _canonical_matrix(df: pd.DataFrame, spec: list[tuple[str, list[str]]]) -> tuple[torch.Tensor, list[str]]:
    cols = []
    values = []
    for canonical, aliases in spec:
        hit = next((a for a in aliases if a in df.columns and pd.api.types.is_numeric_dtype(df[a])), None)
        cols.append(canonical)
        if hit is None:
            values.append(np.zeros(len(df), dtype=np.float32))
        else:
            values.append(pd.to_numeric(df[hit], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32))
    arr = np.column_stack(values).astype(np.float32, copy=False) if values else np.zeros((len(df), 0), np.float32)
    return torch.from_numpy(arr), cols


def _build_edge_index(df: pd.DataFrame, src_map: dict[int, int], dst_map: dict[int, int]) -> torch.Tensor:
    src = df["from_node"].map(src_map)
    dst = df["to_node"].map(dst_map)
    valid = src.notna() & dst.notna()
    if not valid.all():
        bad = int((~valid).sum())
        raise ValueError(f"Edge index contains {bad} node ids absent from node tables")
    return torch.tensor(np.vstack([src.astype(int), dst.astype(int)]), dtype=torch.long)


def _canonicalize_edge_dynamic(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for src, dst in [
        ("1d_edge_flow", "flow"), ("1d_edge_velocity", "velocity"),
        ("2d_flow", "flow"), ("2d_velocity", "velocity")
    ]:
        if src in df.columns and dst not in df.columns:
            rename[src] = dst
    return df.rename(columns=rename)


@dataclass
class StaticGraph:
    model_id: str
    node1_static: torch.Tensor
    node2_static: torch.Tensor
    node1_ids: torch.Tensor
    node2_ids: torch.Tensor
    edge1_index: torch.Tensor
    edge2_index: torch.Tensor
    coupling_index: torch.Tensor  # [2, E], row0 1D local idx, row1 2D local idx
    edge1_static: torch.Tensor
    edge2_static: torch.Tensor
    node1_feature_names: list[str]
    node2_feature_names: list[str]
    edge1_feature_names: list[str]
    edge2_feature_names: list[str]
    edge1_ids: torch.Tensor
    edge2_ids: torch.Tensor

    @property
    def n1(self) -> int:
        return self.node1_static.shape[0]

    @property
    def n2(self) -> int:
        return self.node2_static.shape[0]

    def to(self, device: torch.device | str) -> "StaticGraph":
        return StaticGraph(
            self.model_id, self.node1_static.to(device), self.node2_static.to(device),
            self.node1_ids.to(device), self.node2_ids.to(device),
            self.edge1_index.to(device), self.edge2_index.to(device), self.coupling_index.to(device),
            self.edge1_static.to(device), self.edge2_static.to(device),
            self.node1_feature_names, self.node2_feature_names, self.edge1_feature_names, self.edge2_feature_names,
            self.edge1_ids.to(device), self.edge2_ids.to(device),
        )


@dataclass
class EventData:
    model_id: str
    event_id: str
    node1: torch.Tensor      # [T,N1,2] water_level,inlet_flow
    node2: torch.Tensor      # [T,N2,3] rainfall,water_level,water_volume
    edge1: torch.Tensor      # [T,E1,2] flow,velocity
    edge2: torch.Tensor      # [T,E2,2] flow,velocity
    timestamps: list[str]

    @property
    def timesteps(self) -> int:
        return self.node1.shape[0]

    def to(self, device: torch.device | str) -> "EventData":
        return EventData(self.model_id, self.event_id, self.node1.to(device), self.node2.to(device),
                         self.edge1.to(device), self.edge2.to(device), self.timestamps)


@dataclass(frozen=True)
class EventRef:
    model_id: str
    split: str
    event_id: str
    path: Path


class UrbanFloodBenchRepository:
    """Strict, leakage-aware reader for the machine-learning-ready UrbanFloodBench CSV release."""

    def __init__(self, root: str | Path, model_glob: str = "Model_*", strict_schema: bool = True):
        self.root = Path(root)
        self.model_glob = model_glob
        self.strict_schema = strict_schema
        if not self.root.exists():
            raise FileNotFoundError(f"UrbanFloodBench root does not exist: {self.root}")

    def model_dirs(self) -> list[Path]:
        models = sorted(p for p in self.root.glob(self.model_glob) if p.is_dir())
        if not models:
            raise FileNotFoundError(f"No model directories matching {self.model_glob} under {self.root}")
        return models

    def list_events(self, split: str = "train") -> list[EventRef]:
        refs: list[EventRef] = []
        for model in self.model_dirs():
            split_dir = model / split
            if not split_dir.exists():
                continue
            if self.strict_schema:
                validate_split_schema(split_dir, strict_event_edges=(split == "train")).raise_if_invalid()
            for event in sorted(p for p in split_dir.iterdir() if p.is_dir() and p.name.lower().startswith("event")):
                refs.append(EventRef(model.name, split, event.name, event))
        return refs

    def load_static(self, model_id: str, split: str = "train") -> StaticGraph:
        split_dir = self.root / model_id / split
        files = resolve_static_files(split_dir)
        required = {"nodes1", "nodes2", "index1", "index2", "coupling"}
        missing = required - set(files)
        if missing:
            raise FileNotFoundError(f"{model_id}/{split}: missing static file groups {sorted(missing)}")
        n1df = pd.read_csv(files["nodes1"])
        n2df = pd.read_csv(files["nodes2"])
        n1_ids = n1df["node_idx"].astype(int).to_numpy()
        n2_ids = n2df["node_idx"].astype(int).to_numpy()
        if len(np.unique(n1_ids)) != len(n1_ids) or len(np.unique(n2_ids)) != len(n2_ids):
            raise ValueError("node_idx must be unique within each node type")
        n1_map = {int(v): i for i, v in enumerate(n1_ids)}
        n2_map = {int(v): i for i, v in enumerate(n2_ids)}
        e1df = pd.read_csv(files["index1"])
        e2df = pd.read_csv(files["index2"])
        e1_idx = _build_edge_index(e1df, n1_map, n1_map)
        e2_idx = _build_edge_index(e2df, n2_map, n2_map)
        cdf = pd.read_csv(files["coupling"])
        s = cdf["node_1d"].map(n1_map)
        d = cdf["node_2d"].map(n2_map)
        if s.isna().any() or d.isna().any():
            raise ValueError("1D-2D coupling references unknown node ids")
        coupling = torch.tensor(np.vstack([s.astype(int), d.astype(int)]), dtype=torch.long)

        n1x, n1_names = _canonical_matrix(n1df, NODE1_STATIC_SPEC)
        n2x, n2_names = _canonical_matrix(n2df, NODE2_STATIC_SPEC)
        # Absolute map coordinates are arbitrary across cities/CRSs. Encode them in a city-local,
        # dimensionless frame to avoid a spurious cross-city domain identifier.
        for cname in ("position_x", "position_y"):
            j = n2_names.index(cname)
            col = n2x[:, j]
            span = (col.max() - col.min()).clamp_min(1.0)
            n2x[:, j] = (col - col.mean()) / span

        if "edges1" in files:
            a1df = pd.read_csv(files["edges1"]).sort_values("edge_idx") if "edge_idx" in pd.read_csv(files["edges1"], nrows=1).columns else pd.read_csv(files["edges1"])
            a1x, a1_names = _canonical_matrix(a1df, EDGE1_STATIC_SPEC)
        else:
            a1x, a1_names = torch.zeros(e1_idx.shape[1], len(EDGE1_STATIC_SPEC)), [x[0] for x in EDGE1_STATIC_SPEC]
        if "edges2" in files:
            a2df = pd.read_csv(files["edges2"]).sort_values("edge_idx") if "edge_idx" in pd.read_csv(files["edges2"], nrows=1).columns else pd.read_csv(files["edges2"])
            a2x, a2_names = _canonical_matrix(a2df, EDGE2_STATIC_SPEC)
        else:
            a2x, a2_names = torch.zeros(e2_idx.shape[1], len(EDGE2_STATIC_SPEC)), [x[0] for x in EDGE2_STATIC_SPEC]
        if a1x.shape[0] not in (0, e1_idx.shape[1]):
            raise ValueError("1D edge static rows do not align with edge index")
        if a2x.shape[0] not in (0, e2_idx.shape[1]):
            raise ValueError("2D edge static rows do not align with edge index")
        e1_ids = torch.arange(e1_idx.shape[1], dtype=torch.long)
        e2_ids = torch.arange(e2_idx.shape[1], dtype=torch.long)
        if "edges1" in files and "edge_idx" in a1df:
            e1_ids = torch.tensor(a1df["edge_idx"].to_numpy(np.int64))
        if "edges2" in files and "edge_idx" in a2df:
            e2_ids = torch.tensor(a2df["edge_idx"].to_numpy(np.int64))
        return StaticGraph(
            model_id, n1x.float(), n2x.float(), torch.tensor(n1_ids), torch.tensor(n2_ids),
            e1_idx, e2_idx, coupling, a1x.float(), a2x.float(), n1_names, n2_names,
            a1_names, a2_names, e1_ids, e2_ids,
        )

    def load_event(self, ref: EventRef, graph: StaticGraph | None = None) -> EventData:
        graph = graph or self.load_static(ref.model_id, ref.split)
        cache = ref.path / "hydrograph_event.pt"
        if cache.exists():
            payload = torch.load(cache, map_location="cpu", weights_only=False)
            if payload.get("contract") == "HYDROGRAPH_EVENT_V1":
                return payload["event"]
        files = resolve_event_files(ref.path)
        needed = {"nodes1", "nodes2", "timesteps"}
        missing = needed - set(files)
        if missing:
            raise FileNotFoundError(f"{ref.path}: missing event file groups {sorted(missing)}")
        tsdf = pd.read_csv(files["timesteps"])
        T = len(tsdf)
        if "timestep" in tsdf.columns:
            tvals = tsdf["timestep"].astype(int).to_numpy()
            if not np.array_equal(tvals, np.arange(T)):
                raise ValueError(f"{ref.event_id}: timesteps.csv must be contiguous 0..T-1")
        n1_map = {int(v): i for i, v in enumerate(graph.node1_ids.tolist())}
        n2_map = {int(v): i for i, v in enumerate(graph.node2_ids.tolist())}
        e1_map = {int(v): i for i, v in enumerate(graph.edge1_ids.tolist())}
        e2_map = {int(v): i for i, v in enumerate(graph.edge2_ids.tolist())}

        def node_tensor(path: Path, mapping: dict[int, int], cols: list[str], n: int) -> torch.Tensor:
            df = pd.read_csv(path)
            arr = np.full((T, n, len(cols)), np.nan, dtype=np.float32)
            for t, grp in df.groupby("timestep", sort=False):
                t = int(t)
                if not 0 <= t < T:
                    raise ValueError(f"{path.name}: timestep {t} outside 0..{T-1}")
                idx = grp["node_idx"].map(mapping)
                if idx.isna().any():
                    raise ValueError(f"{path.name}: node_idx not found in static node table")
                arr[t, idx.astype(int).to_numpy(), :] = grp[cols].to_numpy(np.float32)
            if np.isnan(arr).any():
                raise ValueError(f"{path.name}: incomplete node×timestep coverage")
            return torch.from_numpy(arr)

        def edge_tensor(path: Path | None, mapping: dict[int, int], n: int) -> torch.Tensor:
            if path is None:
                return torch.zeros(T, n, 2, dtype=torch.float32)
            df = _canonicalize_edge_dynamic(pd.read_csv(path))
            if "flow" not in df or "velocity" not in df:
                raise ValueError(f"{path.name}: expected flow/velocity dynamic columns")
            arr = np.full((T, n, 2), np.nan, dtype=np.float32)
            for t, grp in df.groupby("timestep", sort=False):
                idx = grp["edge_idx"].map(mapping)
                if idx.isna().any():
                    # Some releases use implicit 0..E-1 indices even if static edge_idx differs.
                    if grp["edge_idx"].min() >= 0 and grp["edge_idx"].max() < n:
                        idx = grp["edge_idx"].astype(int)
                    else:
                        raise ValueError(f"{path.name}: edge_idx not found in static edge table")
                arr[int(t), idx.astype(int).to_numpy(), :] = grp[["flow", "velocity"]].to_numpy(np.float32)
            if np.isnan(arr).any():
                raise ValueError(f"{path.name}: incomplete edge×timestep coverage")
            return torch.from_numpy(arr)

        node1 = node_tensor(files["nodes1"], n1_map, ["water_level", "inlet_flow"], graph.n1)
        node2 = node_tensor(files["nodes2"], n2_map, ["rainfall", "water_level", "water_volume"], graph.n2)
        edge1 = edge_tensor(files.get("edges1"), e1_map, graph.edge1_index.shape[1])
        edge2 = edge_tensor(files.get("edges2"), e2_map, graph.edge2_index.shape[1])
        if len(tsdf.columns) > 1:
            timestamps = tsdf.iloc[:, -1].astype(str).tolist()
        else:
            timestamps = [str(i) for i in range(T)]
        event = EventData(ref.model_id, ref.event_id, node1.float(), node2.float(), edge1.float(), edge2.float(), timestamps)
        try:
            torch.save({"contract": "HYDROGRAPH_EVENT_V1", "event": event}, cache)
        except OSError:
            pass
        return event

    def iter_events(self, refs: list[EventRef]) -> Iterator[tuple[StaticGraph, EventData]]:
        graphs: dict[tuple[str, str], StaticGraph] = {}
        for ref in refs:
            key = (ref.model_id, ref.split)
            if key not in graphs:
                graphs[key] = self.load_static(ref.model_id, ref.split)
            yield graphs[key], self.load_event(ref, graphs[key])
