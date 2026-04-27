# ENS 492 - MatchMaker Pancreatic Synergy Project

This repository contains our MatchMaker-based workflow for pancreatic drug-combination synergy prediction.  
It includes both a reproducible script pipeline and multiple exploratory notebooks used during development.

## Project Scope

We compare model behavior across:
- two dataset variants: `unfiltered` and `disagreement_filtered`,
- five split settings: `LTO`, `LPO`, `LCO`, `LODO`, `LDO`,
- three seeds: `42`, `43`, `44`.

Total runs: `2 x 5 x 3 = 30`.

---

## Pipeline Summary

### 1) Disagreement-aware preprocessing
Script: `scripts/prepare_pancreatic_data.py`

Input:
- `data/synergy - comb - Combination data.csv`

Operations:
- canonicalizes pair order using `(sorted(drug1_id, drug2_id), cell_line)` grouping,
- aggregates replicate Bliss values into `synergy_loewe` (mean),
- builds `synergy_binary` via majority vote,
- marks mixed replicate groups with `disagreement_flag = 1`,
- creates filtered set where `disagreement_flag == 0`.

Outputs:
- `data/processed/pancreatic_unfiltered.tsv`
- `data/processed/pancreatic_disagreement_filtered.tsv`
- `data/processed/data_card.md`

### 2) Split generation
Script: `scripts/generate_splits.py`

Creates seed-specific train/val/test indices for:
- `lto`: random sample split,
- `lpo`: leave pair out,
- `lco`: leave cell line out,
- `lodo`: strict one-unseen-drug split,
- `ldo`: strict both-unseen-drugs split.

Outputs:
- `splits/<split_name>/seed_<seed>/{train_inds.txt,val_inds.txt,test_inds.txt}`
- `splits/leakage_report.json`

### 3) Training/evaluation matrix
Script: `scripts/run_experiments.py`

For each dataset + split + seed combination, calls `main.py` and stores per-run predictions/metrics.

### 4) Reporting
Script: `scripts/build_report.py`

Builds markdown report and reads:
- `results/per_split_seed_metrics.csv`
- `results/summary_metrics.csv`

---

## Model Details

Core model file: `MatchMaker.py`  
Driver script: `main.py`

Architecture (`architecture.txt`):
- `DSN_1 = 2048-4096-2048`
- `DSN_2 = 2048-4096-2048`
- `SPN = 2048-1024`

Training settings (`main.py`):
- loss: weighted MSE,
- optimizer: Adam (`lr=1e-4`, `beta1=0.9`, `beta2=0.999`, `clipnorm=1.0`),
- input dropout: `0.2`,
- hidden dropout: `0.5`,
- batch size: `128`,
- max epochs: `1000`,
- early stopping patience: `100`,
- callbacks: `ModelCheckpoint`, `EarlyStopping`, `TerminateOnNaN`.

Evaluation:
- regression: `MSE`, `Spearman`, `Pearson`,
- classification: `AUC`, `AUPRC`, `F1`,
- prediction is averaged across both orders: `(drug1,drug2)` and `(drug2,drug1)`.

---

## Notebook Guide (Important)

This repo has several notebooks with different roles:

- `colab_run_matchmaker.ipynb` (**primary notebook**)  
  End-to-end Colab runner for the current pipeline (`prepare -> splits -> run_experiments -> report`).

- `uniquq.ipynb` (**feature engineering / preprocessing scratch notebook**)  
  Builds alternative RDKit-based features and custom matrices (`drug_chem_rdkit*`, `cell_line_gex_new.csv`, etc.).
  Not required for the current 30-run script pipeline.

- `preliminary.ipynb` (**exploratory notebook**)  
  Early experiments and intermediate checks.

Recommendation:
- Use `colab_run_matchmaker.ipynb` or the `scripts/*.py` workflow for reproducible results.
- Treat `uniquq.ipynb`, `preliminary.ipynb`, and `bitirme_meeting1.ipynb` as exploratory/legacy unless needed.

---

## Current Repository Layout

```text
.
├── main.py
├── MatchMaker.py
├── helper_funcs.py
├── performance_metrics.py
├── architecture.txt
├── scripts/
│   ├── prepare_pancreatic_data.py
│   ├── generate_splits.py
│   ├── run_experiments.py
│   └── build_report.py
├── data/
│   ├── DrugCombinationData.tsv
│   ├── cell_line_gex.csv
│   ├── drug1_chem.csv
│   ├── drug2_chem.csv
│   └── processed/
├── splits/
├── results/
├── colab_run_matchmaker.ipynb
├── bitirme_meeting1.ipynb
├── uniquq.ipynb
└── preliminary.ipynb
```

---

## Setup

### 1) Create environment
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2) Install dependencies
If you have `requirements.txt`:
```bash
pip install -r requirements.txt
```

Otherwise:
```bash
pip install numpy pandas scipy scikit-learn tensorflow matplotlib seaborn h5py
```

---

## Reproducible Run (Script Pipeline)

Run these from repo root:

### Step 1 - Build processed datasets
```bash
.venv/bin/python scripts/prepare_pancreatic_data.py
```

### Step 2 - Generate split indices
```bash
.venv/bin/python scripts/generate_splits.py \
  --input-tsv data/processed/pancreatic_disagreement_filtered.tsv \
  --outdir splits \
  --seeds 42 43 44
```

### Step 3 - Run full experiment matrix
```bash
.venv/bin/python scripts/run_experiments.py \
  --project-root . \
  --seeds 42 43 44 \
  --splits-root splits \
  --processed-root data/processed \
  --out-root results/runs \
  --classification-label-column synergy_binary \
  --classification-threshold 0.0
```

### Step 4 - Build report
```bash
.venv/bin/python scripts/build_report.py \
  --summary-csv results/summary_metrics.csv \
  --per-seed-csv results/per_split_seed_metrics.csv \
  --out-md results/report.md
```

Optional command validation:
```bash
.venv/bin/python scripts/run_experiments.py --dry-run
```

---

## CPU-only vs GPU

If running on CPU-only hardware, keep using notebooks or scripts with CPU settings (slower but valid).

Example CPU-oriented run:
```bash
.venv/bin/python scripts/run_experiments.py \
  --project-root . \
  --seeds 42 43 44 \
  --splits-root splits \
  --processed-root data/processed \
  --out-root results/runs \
  --gpu-devices ""
```

---

## Key Outputs

- Processed data:
  - `data/processed/pancreatic_unfiltered.tsv`
  - `data/processed/pancreatic_disagreement_filtered.tsv`
  - `data/processed/data_card.md`
- Split files:
  - `splits/<split>/seed_<seed>/train_inds.txt`
  - `splits/<split>/seed_<seed>/val_inds.txt`
  - `splits/<split>/seed_<seed>/test_inds.txt`
  - `splits/leakage_report.json`
- Metrics/reports:
  - `results/per_split_seed_metrics.csv`
  - `results/summary_metrics.csv`
  - `results/report.md`

---

## Notes and Conventions

- `synergy_loewe` is used as the regression label column name for compatibility with the training pipeline; in this project it stores mean-aggregated Bliss.
- Seeds are used to reduce random variance and report stable mean performance.
- Keep architecture and hyperparameters fixed across split settings for fair comparison.

---

## References

- MatchMaker paper: [bioRxiv](https://www.biorxiv.org/content/10.1101/2020.05.24.113241v3)
- DrugComb portal: [drugcomb.fimm.fi](https://drugcomb.fimm.fi/)

## License

[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
