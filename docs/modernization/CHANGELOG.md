# Modernization Changelog

All changes made during repository modernization are documented here.

## [Unreleased]

### P0-1: Ignore env/logs/checkpoints and stop tracking binaries
- Updated `.gitignore` with stricter patterns
- Removed tracked checkpoints from git index (85 MB freed from tracking)
- Removed tracked lightning_logs from git index (27 MB freed from tracking)
- Removed large binary models from git index (keeps scaler.pkl and CSV)
- **Why**: Binary files bloat repository size and slow down clones

### P0-2: Strip notebook outputs with nbstripout + pre-commit
- Added `.pre-commit-config.yaml` with nbstripout hook
- Added `.gitattributes` for automatic notebook filtering
- **Why**: Notebook outputs create large diffs and merge conflicts

### P0-3: Document env as source of truth
- Updated `environment.yml` with all required dependencies
- Added header comment to `requirements.txt` explaining it's derived
- Added `scripts/export_requirements.sh` for generating requirements.txt
- **Why**: Single source of truth prevents dependency drift

### P1-1: Move FFT feature extraction to src/features
- Created `src/features.py` with extracted functions
- Functions: `extract_time_features`, `zero_crossing_rate`, `fft_features`
- Updated notebook 03 to import from `src.features`
- **Why**: Reusable code, testable, inference consistency

### P1-2: Add smoke tests for features and model
- Created `tests/test_imports.py` for import verification
- Created `tests/test_smoke.py` for model and feature tests
- Added `make test` target to Makefile
- **Why**: Catch regressions early, verify installation

### P1-3: Add CLI inference script
- Created `scripts/predict.py` for batch/single inference
- Created `docs/INFERENCE_GUIDE.md` with usage examples
- **Why**: Production-ready inference pipeline
