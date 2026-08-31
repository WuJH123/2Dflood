from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def write_model(root: Path, model_id: int, events: int, steps: int, seed: int) -> None:
    rng = np.random.default_rng(seed + model_id)
    base = root / f"Model_{model_id}" / "train"
    base.mkdir(parents=True, exist_ok=True)
    n1, n2 = 3, 8
    e1 = [(0, 1), (1, 2)]
    e2 = [(i, i + 1) for i in range(n2 - 1)] + [(i + 1, i) for i in range(n2 - 1)]
    cpl = [(0, 1), (1, 4), (2, 6)]

    pd.DataFrame({
        "node_idx": np.arange(n1), "depth": [8, 9, 10], "invert_elevation": [90, 89, 88],
        "surface_elevation": [100, 99, 98], "base_area": [20, 20, 20],
    }).to_csv(base / "1d_nodes_static.csv", index=False)
    pd.DataFrame({
        "node_idx": np.arange(n2), "2d_position_x": np.arange(n2) * 20.0,
        "2d_position_y": np.zeros(n2), "area": np.full(n2, 400.0), "roughness": np.full(n2, 0.04),
        "min_elevation": 95 + np.linspace(0, 1, n2), "elevation": 95 + np.linspace(0, 1, n2),
        "aspect": np.full(n2, 90.0), "curvature": np.zeros(n2), "flow_accumulation": np.arange(n2),
    }).to_csv(base / "2d_nodes_static.csv", index=False)
    pd.DataFrame({"from_node": [x[0] for x in e1], "to_node": [x[1] for x in e1]}).to_csv(base / "1d_edge_index.csv", index=False)
    pd.DataFrame({"from_node": [x[0] for x in e2], "to_node": [x[1] for x in e2]}).to_csv(base / "2d_edge_index.csv", index=False)
    pd.DataFrame({"node_1d": [x[0] for x in cpl], "node_2d": [x[1] for x in cpl]}).to_csv(base / "1d2d_connections.csv", index=False)
    pd.DataFrame({
        "edge_idx": np.arange(len(e1)), "relative_position_x": [20, 20], "relative_position_y": [0, 0],
        "length": [100, 100], "diameter": [3, 3], "shape": [1, 1], "roughness": [0.013, 0.013], "slope": [0.01, 0.01],
    }).to_csv(base / "1d_edges_static.csv", index=False)
    pd.DataFrame({
        "edge_idx": np.arange(len(e2)), "relative_position_x": np.tile([20, -20], len(e2)//2 + 1)[:len(e2)],
        "relative_position_y": np.zeros(len(e2)), "face_length": np.full(len(e2), 20),
        "length": np.full(len(e2), 20), "slope": np.tile([0.002, -0.002], len(e2)//2 + 1)[:len(e2)],
    }).to_csv(base / "2d_edges_static.csv", index=False)

    elev = 95 + np.linspace(0, 1, n2)
    area = 400.0
    for ev in range(events):
        edir = base / f"event_{ev+1}"
        edir.mkdir(exist_ok=True)
        pd.DataFrame({"timestep": np.arange(steps), "timestamp": pd.date_range("2026-01-01", periods=steps, freq="5min")}).to_csv(edir / "timesteps.csv", index=False)
        rain = np.maximum(0, np.sin(np.linspace(-1, 3.2, steps)) * (0.04 + 0.01 * ev))
        depth = np.zeros((steps, n2), np.float32)
        for t in range(1, steps):
            depth[t] = np.maximum(0, 0.90 * depth[t-1] + rain[t] / 12.0 * (0.6 + np.linspace(0.5, 1.0, n2)))
        volume = depth * area
        wl2 = elev[None, :] + depth
        rows2 = []
        for t in range(steps):
            for i in range(n2):
                rows2.append((t, i, rain[t], wl2[t, i], volume[t, i]))
        pd.DataFrame(rows2, columns=["timestep", "node_idx", "rainfall", "water_level", "water_volume"]).to_csv(edir / "2d_nodes_dynamic_all.csv", index=False)

        inlet = np.stack([depth[:, 1], depth[:, 4], depth[:, 6]], axis=1) * 0.4
        wl1 = np.array([100, 99, 98])[None, :] + inlet * 0.05
        rows1 = []
        for t in range(steps):
            for i in range(n1):
                rows1.append((t, i, wl1[t, i], inlet[t, i]))
        pd.DataFrame(rows1, columns=["timestep", "node_idx", "water_level", "inlet_flow"]).to_csv(edir / "1d_nodes_dynamic_all.csv", index=False)

        rows_e1 = []
        for t in range(steps):
            for j in range(len(e1)):
                q = float(0.1 * (inlet[t, j] - inlet[t, j+1]))
                rows_e1.append((t, j, q, q / 7.0))
        pd.DataFrame(rows_e1, columns=["timestep", "edge_idx", "flow", "velocity"]).to_csv(edir / "1d_edges_dynamic_all.csv", index=False)
        rows_e2 = []
        for t in range(steps):
            for j, (a, b) in enumerate(e2):
                q = float((depth[t, a] - depth[t, b]) * 0.5)
                rows_e2.append((t, j, q, q / 20.0))
        pd.DataFrame(rows_e2, columns=["timestep", "edge_idx", "flow", "velocity"]).to_csv(edir / "2d_edges_dynamic_all.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/synthetic_urbanfloodbench")
    ap.add_argument("--models", type=int, default=3)
    ap.add_argument("--events", type=int, default=4)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    root = Path(args.output)
    for i in range(1, args.models + 1):
        write_model(root, i, args.events, args.steps, args.seed)
    print(root.resolve())


if __name__ == "__main__":
    main()
