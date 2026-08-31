# Formal runbook and operating notes

This runbook is the operational contract for reproducing the UrbanFloodBench experiments. Read it before launching an expensive training run.

## 1. Environment

Recommended baseline:

- Python 3.10 or 3.11 for formal runs.
- PyTorch >= 2.2 with a CUDA build matched to the installed NVIDIA driver when a GPU is used.
- A clean virtual environment per experiment series.
- Enough local storage for the official UrbanFloodBench release, checkpoints and event-level evidence. The repository intentionally does not vendor data or weights.

Create the environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install the desired CUDA/CPU PyTorch build first if needed.
pip install -e ".[dev,plot]"
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the desired CUDA/CPU PyTorch build first if needed.
pip install -e '.[dev,plot]'
```

Do not treat `pytest` before the editable install as a repository failure: the package lives under `src/` and must be installed, or `PYTHONPATH=src` must be supplied for a developer-only source-tree check.

## 2. Data placement and immutable raw data

Set `data.root` in `configs/paper.yaml` to the parent directory containing the UrbanFloodBench `Model_*` folders. Keep the downloaded official data read-only whenever possible. Do not rename node IDs, delete rows, fill missing dynamic records manually, or otherwise modify the raw release to make the loader pass.

Expected high-level layout:

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

The official research release and competition mirrors may differ slightly. Resolve such differences in `hydrograph.data`/`hydrograph.schema`, never by silently altering scientific data.

## 3. Mandatory pre-training audit

Run the schema and scientific audit before any formal training:

```bash
hydrograph inspect --config configs/paper.yaml
hydrograph audit \
  --config configs/paper.yaml \
  --split train \
  --full \
  --fail-on-p1 \
  --output outputs/data_audit.json
```

Stop the formal run if a P0/P1 finding remains. P0 includes target-city leakage, future hydraulic truth leakage, incorrect node/edge/timestep alignment, or invalid physics/units. P1 includes split/normalization/checkpoint lineage errors that can materially change paper conclusions.

## 4. Formal split contract

The main paper protocol is leave-one-city-out (LOCO):

- Complete rainfall/flood events are the split unit; never split random timesteps across train/validation/test.
- The target city is excluded from source training and source normalization statistics.
- Target adaptation events are disjoint from target evaluation events.
- Hyperparameter selection must not use held-out target evaluation events.
- Save `split_manifest.json`, `resolved_config.json` and the normalization lineage with every formal checkpoint.

## 5. Source-city generalist training

Example:

```bash
hydrograph train \
  --config configs/paper.yaml \
  --target-model Model_4 \
  --output-dir outputs/loco_Model_4
```

Before a long run, first execute a short smoke configuration and confirm:

- loss is finite;
- no negative-volume/NaN instability is generated;
- GPU memory remains bounded over an autoregressive rollout;
- checkpoints can be reloaded and reproduce validation outputs;
- target-city data do not appear in normalization or source manifests.

## 6. Target-city evaluation

```bash
hydrograph paper-protocol \
  --config configs/paper.yaml \
  --checkpoint outputs/loco_Model_4/best.pt \
  --target-model Model_4 \
  --output-dir outputs/loco_Model_4/paper_evidence
```

The formal target forecast is autoregressive after the declared warm-up. Future rainfall may be used because it is the prescribed forcing in this benchmark; future hydraulic states may not be used.

## 7. Sparse-city contract

Sparse adaptation is deliberately stricter than ordinary fine-tuning. The target city may provide:

- static geometry/topology;
- known rainfall forcing;
- historical water level only at the declared sensor nodes.

It must not provide hidden target water levels, water-volume truth, inlet-flow truth, edge-flow truth or velocity truth to the sparse adaptation loss/state update.

Example:

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

For the formal paper, repeat sensor layouts with multiple seeds and report mean ± standard deviation. Record the exact sensor node IDs.

## 8. Ablations and runtime

Mechanism ablations:

```bash
python scripts/run_ablations.py --config configs/paper.yaml --target-model Model_4
```

Runtime benchmark:

```bash
python scripts/benchmark_runtime.py \
  --config configs/paper.yaml \
  --checkpoint outputs/loco_Model_4/best.pt \
  --model-id Model_4
```

Do not report a simulator speedup unless the numerical-simulator runtime was measured on a clearly documented hardware/protocol basis.

## 9. Tests and quality gate

Normal developer/CI route after installation:

```bash
python -m compileall -q src scripts tests
pytest -q
```

Source-tree fallback when package installation is intentionally skipped:

```bash
PYTHONPATH=src pytest -q
```

On Windows PowerShell the equivalent fallback is:

```powershell
$env:PYTHONPATH = "src"
pytest -q
```

The synthetic generator is for software verification only:

```bash
python scripts/make_synthetic_urbanfloodbench.py --output data/synthetic
pytest -q
```

Never use synthetic metrics as paper evidence.

## 10. Evidence to archive for each formal run

Archive at minimum:

- git commit SHA;
- complete resolved configuration;
- raw-data release identifier/checksum when available;
- split/event manifest;
- source-only normalization lineage;
- random seeds and sensor IDs;
- best/last checkpoint identifiers;
- event-balanced metrics;
- ablation configuration;
- hardware/software versions and runtime logs;
- data/schema audit output.

A result without this lineage should be considered development-only rather than formal paper evidence.

## 11. Common failure modes

1. **Package import failure** — install with `pip install -e .` before running tests/CLI.
2. **CUDA mismatch** — install the PyTorch build matched to the local driver; this repository does not pin a CUDA wheel URL.
3. **Schema mismatch** — run `hydrograph inspect/audit`; fix aliases or loader logic, not raw data.
4. **Out-of-memory rollout** — reduce training window/batch settings for development, then re-register the formal configuration; do not silently alter the locked experiment.
5. **Target leakage** — inspect split and normalization manifests; sparse target adaptation must remain water-level-only.
6. **Misleading generalization claim** — two Kaggle domains alone do not establish broad cross-city generalization; use the complete research release and/or an external city for a stronger claim.
