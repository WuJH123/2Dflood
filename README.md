# 2Dflood — HydroGraph-Operator

Research code for a **transferable, physics-constrained 1D–2D urban flood-field surrogate** built around the public **UrbanFloodBench** dataset. The repository is designed for a Water Research-style study of three questions:

1. Can hydraulic interactions learned in source cities transfer to an unseen city?
2. Do explicit surface–sewer topology, flux supervision and mass conservation improve out-of-city generalization?
3. How much target-city information is required when only sparse water-level sensors are available?

> Status: research implementation and verification suite. The code has synthetic end-to-end tests, but **no scientific performance claim is made until the official UrbanFloodBench data are run under the locked protocols below**.

## Scientific design

The model represents each city as a heterogeneous hydraulic graph

`G = G_surface ⊕ G_sewer ⊕ G_exchange`

with 1D drainage nodes/links, 2D surface cells/links, and 1D–2D exchange connections. It jointly predicts node states and edge fluxes. Key components are:

- **Hydraulic directional message passing** using water-level/head differences, topology, terrain/pipe attributes and dynamic edge states.
- **Physics-token global mixing**: nodes are softly grouped into latent hydraulic-state tokens, and global attention is performed among tokens rather than all mesh points. This is inspired by the physics-state idea of Transolver, implemented independently for heterogeneous urban hydraulics.
- **Flux/state co-prediction**: 1D/2D edge flow and velocity are learned together with node water level/volume.
- **Mass-conservation losses** in native UrbanFloodBench units. Local surface balance uses rainfall, 2D flux divergence and aggregate 1D inlet exchange; global surface balance is reported separately.
- **Masked hydraulic pretraining** to teach the source-city model to reconstruct dynamics from incomplete state information.
- **Sparse-city adaptation** using only target-city water-level sensors plus known rainfall. Hidden target fields, edge truth and volume truth are not used by the adaptation loss.
- **Parameter-efficient adapters** with a frozen generalist backbone.

The implementation intentionally does **not** copy Transolver/HAMLET/LaMO/CurvGT source code. Their ideas inform the design, while the code here is written specifically around UrbanFloodBench's coupled 1D–2D data contract.

## Dataset

Use the official UrbanFloodBench research release where possible. The dataset page reports four coupled 1D–2D domains — Beaver Lake, Davis, New Orleans and Coogee — generated with HEC-RAS 6.7 Beta 4a. The Kaggle competition used Beaver Lake and Davis and exposes the machine-learning-ready CSV schema.

Expected layout:

```text
<data_root>/
  Model_*/
    train/
      1d_nodes_static.csv
      2d_nodes_static.csv
      1d_edges_static.csv
      2d_edges_static.csv
      1d_edge_index.csv
      2d_edge_index.csv
      1d2d_connections.csv
      event_*/
        timesteps.csv
        1d_nodes_dynamic_all.csv
        2d_nodes_dynamic_all.csv
        1d_edges_dynamic_all.csv
        2d_edges_dynamic_all.csv
```

The loader accepts minor filename/column aliases but fails closed on node/edge alignment, incomplete timestep coverage, non-finite data, and unknown IDs.

**Dataset license:** the University of Sydney UrbanFloodBench research release is listed as CC BY-NC 4.0. This repository does not redistribute the dataset. Check the license of the exact mirror you download before redistribution or commercial use.

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

PyTorch should be installed with the CUDA build appropriate for the local machine before/alongside the editable install.

## 1. Audit the official data before training

Set `data.root` in `configs/paper.yaml`, then run:

```bash
hydrograph inspect --config configs/paper.yaml
hydrograph audit --config configs/paper.yaml --split train --full --fail-on-p1 \
  --output outputs/data_audit.json
```

Do not start formal training if the audit raises a P0 or P1 finding.

## 2. Leave-one-city-out training

Example holding out one city/model domain:

```bash
hydrograph train \
  --config configs/paper.yaml \
  --target-model Model_4 \
  --output-dir outputs/loco_Model_4
```

The training code writes:

- `split_manifest.json`: exact event partition;
- `resolved_config.json`: frozen run configuration;
- `train.jsonl`: epoch metrics;
- `best.pt`, `last.pt`, periodic checkpoints;
- source-only normalization lineage in the checkpoint.

For LOCO experiments, the held-out city is forbidden from source training **and normalization-statistic fitting**.

## 3. Full target-city paper protocol

```bash
hydrograph paper-protocol \
  --config configs/paper.yaml \
  --checkpoint outputs/loco_Model_4/best.pt \
  --target-model Model_4 \
  --output-dir outputs/loco_Model_4/paper_evidence
```

This evaluates:

- persistence baseline;
- zero-shot target-city prediction with dense warm-up;
- zero-shot sparse-sensor warm-up across sensor fractions/layout seeds;
- event-disjoint few-shot target adaptation;
- sparse adaptation using only sensor water levels;
- event-balanced hydraulic, inundation, timing and mass-balance metrics;
- transfer-recovery statistics.

The formal multi-city driver is:

```bash
python scripts/run_paper_protocol.py --config configs/paper.yaml
```

## 4. Ablation study

```bash
python scripts/run_ablations.py \
  --config configs/paper.yaml \
  --target-model Model_4
```

Available ablations isolate:

- global hydraulic tokens;
- 1D–2D coupling;
- hydraulic-head directionality;
- flux supervision;
- mass-conservation loss;
- masked pretraining.

These correspond directly to the paper's mechanism hypotheses rather than arbitrary architectural knobs.

## 5. Sparse-city adaptation only

```bash
hydrograph adapt \
  --config configs/paper.yaml \
  --checkpoint outputs/loco_Model_4/best.pt \
  --target-model Model_4 \
  --k-events 3 \
  --sensor-fraction 0.01 \
  --sensor-strategy degree_stratified \
  --epochs 20 \
  --output-dir outputs/adapt_Model_4_k3_s001
```

A fixed sensor mask is reused across adaptation/evaluation events to represent a real monitoring network. Adaptation/evaluation events are disjoint.

## 6. Metrics produced for the paper

The evaluator reports event-balanced:

- 1D/2D water-level RMSE, MAE, NSE and R²;
- 2D water-volume error;
- 1D/2D flow and velocity error;
- surface-depth RMSE derived from `water_volume / cell_area`;
- CSI/IoU/F1/precision/recall at 0.05, 0.10, 0.30 and 0.50 m;
- peak-depth error;
- flood arrival-time and duration error;
- local/global surface mass-balance residuals;
- NaN and negative-volume stability diagnostics.

This avoids relying on a single average RMSE and supports hazard-relevant interpretation.

## 7. Causality / leakage contract

P0 scientific rules enforced by code/tests:

- split by **event**; no random-timestep split;
- LOCO target city cannot enter source training or source normalization;
- sparse target adaptation uses only declared sensor water levels and known rainfall;
- unobserved target-city ground truth cannot enter hydraulic-head messages, recurrent state, edge state or residual bases;
- sparse adaptation does not use target full-field/edge labels;
- target adaptation events and target evaluation events are disjoint;
- final target evaluation is autoregressive after the warm-up;
- future rainfall is used only because UrbanFloodBench defines rainfall as a known forcing; future hydraulic truth is never used in forecast rollout.

`tests/test_model.py::test_sparse_assimilation_does_not_read_hidden_truth` explicitly perturbs hidden target fields by a huge amount and verifies the sparse assimilation result is unchanged.

## 8. Runtime benchmark

```bash
python scripts/benchmark_runtime.py \
  --config configs/paper.yaml \
  --checkpoint outputs/loco_Model_4/best.pt \
  --model-id Model_4
```

Report both predictive skill and computational speed relative to the numerical simulator in the manuscript.

## 9. Synthetic smoke test

The repository includes a tiny UrbanFloodBench-compatible generator solely for software verification:

```bash
python scripts/make_synthetic_urbanfloodbench.py --output data/synthetic
pytest
```

Synthetic results are **not scientific evidence**.

## Architecture provenance / recent algorithmic ideas

The design absorbs, but does not reproduce, several recent PDE/graph-learning ideas:

- Wu et al., **Transolver**, ICML 2024 — learned physics-state tokens and geometry-general attention.
- Bryutkin et al., **HAMLET**, ICML 2024 — modular graph-transformer neural operator for arbitrary geometries and limited data.
- Tiwari et al., **Latent Mamba Operator**, ICML 2025 — latent state-space modeling for long-range PDE dynamics.
- Liao et al., **Curvature-aware Graph Attention**, ICML 2025 — intrinsic geometry/curvature as a PDE-relevant inductive bias.
- Acosta et al., **DUALFloodGNN**, 2025/2026 — joint node-volume/edge-flow prediction and mass-conservation guidance for flood GNNs.

The new research hypothesis tested here is not “a Transformer is better than a GNN”; it is that **cross-city transfer emerges from hydraulic connectivity and flux relations that are more invariant than city-specific spatial appearance**, and that this representation can support sparse-city deployment.

## Reproducibility

- deterministic split and run seeds;
- checkpointed config/split/normalization/feature lineage;
- synthetic integration tests for training, target evaluation and sparse adaptation;
- CI executes compilation + tests;
- no dataset or trained weights are committed by default.

See `docs/METHODS_MAPPING.md` for the direct mapping between code modules, paper hypotheses and required result tables/figures, and `docs/RUNBOOK.md` for environment setup, formal-run gates, sparse-city restrictions, evidence archiving and common failure modes.
