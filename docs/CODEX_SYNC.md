# Web GPT → GitHub → Codex synchronization contract

This document defines the intended division of responsibility for the 2Dflood project.

## 1. Authority and responsibilities

- **Web ChatGPT** authors/reviews research code, tests, configs and documentation and synchronizes the accepted version to GitHub.
- **GitHub `main`** is the authoritative code hand-off point.
- **Local Codex** synchronizes the repository and executes it against the user's local software environment and local UrbanFloodBench data.
- **Local scientific data stay local** unless the user explicitly requests upload.

Do not commit virtual environments, downloaded UrbanFloodBench files, checkpoints, caches, or large generated outputs.

## 2. First local checkout

```powershell
git clone https://github.com/WuJH123/2Dflood.git
cd 2Dflood
git switch main
git rev-parse HEAD
```

Create/use the local environment separately and set `data.root` in a local config or the formal project config to the local UrbanFloodBench path.

## 3. Synchronize before every GPT-directed run

PowerShell:

```powershell
cd <LOCAL_2DFLOOD_REPO>
git status --short
git fetch origin
git switch main
git pull --ff-only origin main
git rev-parse HEAD
```

If `git status --short` shows local source-code modifications, Codex must inspect them before pulling. Do not overwrite or auto-merge scientific code silently. Local-only paths such as data, outputs and environments should be ignored by Git.

## 4. Mandatory execution gate after synchronization

```powershell
python -m compileall -q src scripts tests
pytest -q
hydrograph inspect --config configs/paper.yaml
hydrograph audit --config configs/paper.yaml --split train --full --fail-on-p1 --output outputs/data_audit.json
```

Do not launch formal training while a P0/P1 audit finding remains.

## 5. Formal evidence lineage

Every formal run should preserve:

- Git `HEAD` SHA;
- resolved configuration;
- dataset/release identifier or checksum if available;
- event split manifest;
- normalization lineage;
- random seeds and sparse-sensor IDs;
- checkpoint IDs;
- event-balanced metrics;
- hardware/software/runtime information.

## 6. Returning execution findings

When Codex finds a code defect locally, it should report:

1. exact Git SHA reproduced;
2. command used;
3. minimal error/log excerpt;
4. affected file/function;
5. whether the failure is P0/P1/P2/P3;
6. whether any formal scientific result is invalidated.

Web ChatGPT can then revise the repository and push a new `main`, after which Codex re-synchronizes and reruns the relevant regression gate.

## 7. What must not be synchronized to GitHub

Do not upload by default:

- official UrbanFloodBench data;
- `.venv` or Conda environments;
- `outputs/`;
- training checkpoints (`*.pt`, `*.pth`, `*.ckpt`);
- Python caches;
- temporary notebooks/caches containing copied raw data;
- credentials, tokens or machine-specific secrets.

The repository should contain code and reproducibility metadata, not the local research data store.
