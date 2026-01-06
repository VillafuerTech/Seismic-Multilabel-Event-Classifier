# Baseline Snapshot

**Date**: 2026-01-06
**Branch**: `chore/modernize-repo`
**Base Commit**: `5fdb7ff899ac3a0cd3c3a6b4e83e8e522f2f0a32`

## Git Status (Pre-Modernization)

```
On branch chore/modernize-repo
Changes not staged for commit:
    modified:   models/best_seismic-v1.ckpt
    modified:   models/repeated_kfold_scores.csv
    modified:   notebooks/01_Preprocessing.ipynb
    modified:   notebooks/02_Exploratory_Data_Analysis.ipynb
    modified:   notebooks/03_Feature_Engineering.ipynb
    modified:   notebooks/07_Redes_model.ipynb
    modified:   notebooks/08_Model_Comparison.ipynb
```

## Directory Sizes

| Directory        | Size   | Files Tracked in Git |
|------------------|--------|----------------------|
| `models/`        | 30 MB  | 8 files              |
| `checkpoints/`   | 85 MB  | 150 files            |
| `lightning_logs/`| 27 MB  | 157 files            |
| `env/`           | 2.3 GB | 0 files              |

## Tracked Binary Files (To Be Untracked)

### models/ (8 files)
- `best_seismic-v1.ckpt` (792 KB)
- `best_seismic.ckpt` (256 KB)
- `logreg_repkfold.pkl` (13 KB)
- `mlp_repkfold.pkl` (4.7 MB)
- `repeated_kfold_scores.csv` (595 B) - keep tracked (small text file)
- `rf_repkfold.pkl` (25 MB)
- `scaler.pkl` (2.6 KB)
- `svm_repkfold.pkl` (12 KB)

### checkpoints/ (150 files)
- Numerous epoch checkpoints (97KB - 792KB each)
- All should be untracked

### lightning_logs/ (157 files)
- 45 version directories with training logs
- All should be untracked

## Makefile Targets

```makefile
.PHONY: env run-notebooks

# Create Conda environment
env:
    conda env create -f environment.yml

# Execute all notebooks with papermill
run-notebooks:
    mkdir -p outputs
    for NB in notebooks/*.ipynb; do \
      papermill $$NB outputs/$$(basename $${NB%%.ipynb})_output.ipynb; \
    done
```

## Test Status

**No tests exist**. `pytest` is listed in environment.yml but no test files are present.

## Notebooks

| Notebook | Description |
|----------|-------------|
| 01_Preprocessing.ipynb | Data ingestion from .at2 files |
| 02_Exploratory_Data_Analysis.ipynb | EDA and visualization |
| 03_Feature_Engineering.ipynb | FFT computation and features |
| 04_Baseline_Model.ipynb | Baseline classifier |
| 05_SVM_Model.ipynb | SVM with RBF kernel |
| 06_RF_Model.ipynb | Random Forest classifier |
| 07_Redes_model.ipynb | Neural network (PyTorch Lightning) |
| 08_Model_Comparison.ipynb | Model comparison metrics |
| 09_Results_Visualization.ipynb | Results analysis |
| 10_Final_Report_and_Deployment.ipynb | Final report |

## Key Source Files

- `src/seismic_model.py`: PyTorch Lightning model (`SeismicMultilabelModel`)
- `environment.yml`: Conda environment (15 dependencies)
- `requirements.txt`: Pip requirements (17 dependencies, out of sync with environment.yml)

## Environment Discrepancy

**environment.yml** contains:
- numpy, pandas, scipy, scikit-learn, tensorflow, jupyterlab, papermill, joblib, matplotlib, seaborn, pytest

**requirements.txt** adds:
- pyarrow, fastparquet, scikeras, torch, Lightning, torchmetrics

**Action needed**: Consolidate to use environment.yml as source of truth.

## .gitignore Current State

Already ignores:
- `env/`
- `__pycache__/`
- `.ipynb_checkpoints/`
- `data/interim/*`, `data/processed/*`, `data/raw/*` (with exclusions)
- `outputs/`
- `lightning_logs/`
- `checkpoints/`

**Issue**: `checkpoints/` and `lightning_logs/` patterns were added AFTER files were committed.
Need to run `git rm --cached -r` to untrack them.

## Artifacts Required for Inference

1. `models/scaler.pkl` - StandardScaler for feature normalization
2. `models/best_seismic-v1.ckpt` - Best PyTorch Lightning checkpoint
3. Feature extraction logic from notebook 03

## Acceptance Criteria for Modernization

1. [ ] `git status` clean after running notebooks
2. [ ] Notebooks produce same artifact paths
3. [ ] `pytest -q` passes in < 5 seconds
4. [ ] `python scripts/predict.py --help` works
5. [ ] `python scripts/predict.py --input <sample>` runs
