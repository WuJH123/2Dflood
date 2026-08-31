# Paper-to-code mapping

## Scientific hypotheses

| Hypothesis | Code mechanism | Primary ablation/evidence |
|---|---|---|
| H1: explicit surface–sewer topology improves unseen-city transfer | `StaticGraph.coupling_index`, `ProcessorBlock.m12/m21` | `no_coupling` |
| H2: hydraulic directionality and flux learning improve extreme-event stability | head-difference relation messages + edge heads | `no_hydraulic_head`, `no_flux_supervision` |
| H3: global hydraulic-state interactions complement local routing | `HydraulicTokenMixer` | `no_global_tokens` |
| H4: conservation improves physical extrapolation | `surface_mass_residual`, mass losses | `no_mass_loss` |
| H5: masked source-city training enables sparse target deployment | sparse causal assimilation + masked source windows | `no_masked_pretraining` |
| H6: a frozen generalist can adapt from sparse target water-level sensors | residual adapters + sensor-only target loss | few-shot/sensor-fraction curves |

## Proposed Methods sections

### 2.1 UrbanFloodBench and transfer protocol

- `hydrograph.data`
- `hydrograph.schema`
- `hydrograph.split`
- `hydrograph.audit`

Mandatory statement: source/target partitioning is event-wise; LOCO target domains are excluded from source normalization.

### 2.2 Heterogeneous surface–sewer graph

- `StaticGraph`
- relation-specific 1D, 2D and coupling messages in `model.py`

### 2.3 Physics-constrained local–global operator

- local relation messages: `RelationMessage`
- exchange messages: `CouplingMessage`
- global state tokens: `HydraulicTokenMixer`
- temporal recurrent state: processor GRU states
- node and edge decoders: `HydroGraphOperator.forward_step`

### 2.4 Flux/state objective and mass conservation

- `losses.compute_step_loss`
- `physics.surface_mass_residual`

Important limitation to state in the paper: the ML-ready release exposes aggregate 1D-node inlet exchange rather than per-connection exchange flow. Local 2D conservation diagnostics therefore distribute a node's aggregate inlet flow equally across its connected 2D cells; the global exchange term uses the exact aggregate.

### 2.5 Masked pretraining and sparse-city adaptation

- `rollout.assimilate_sparse_history`
- `trainer._sensor_masks`
- `adapt.adapt_sparse_city`

Target sparse adaptation supervises only measured water level. It does not use hidden target water volume, edge flow/velocity or unobserved node labels.

### 2.6 Evaluation

- `metrics.event_metrics`
- `paper_eval.run_target_protocol`
- `scripts/run_ablations.py`
- `scripts/benchmark_runtime.py`

## Required paper figures/tables

1. **Fig. 1**: heterogeneous 1D–2D graph + local flux messages + global hydraulic tokens + sparse adapter.
2. **Fig. 2**: LOCO performance by city; water-level/depth + inundation metrics.
3. **Fig. 3**: sensor fraction / target-event count vs performance; show replicate mean ± SD and transfer-recovery.
4. **Fig. 4**: extreme-event maps and hydrographs; arrival, peak and recession behavior.
5. **Fig. 5**: mechanism ablations and mass-balance diagnostics.
6. **Fig. 6**: matched hydraulic-regime representation analysis (`representation.py`) and runtime speedup.
7. **Table 1**: city/domain statistics and split counts.
8. **Table 2**: full LOCO benchmark against persistence and selected published/reimplemented baselines.
9. **Table 3**: ablations.
10. **Table S1+**: seeds, hyperparameters, sensor IDs, event manifests and complete event-balanced metrics.

## Claims that must not be made from code alone

- “state of the art” before running the complete official benchmark and fair baselines;
- “general across cities” from only the two Kaggle competition domains;
- “physics preserving” solely because a loss is present — report actual mass residuals;
- “data sparse” without an event-disjoint target protocol and explicit sensor fractions;
- “real-world generalization” unless an external/observational city is added beyond synthetic HEC-RAS domains.
