# Repository Modernization Plan

## P0 (Day 1) - Hygiene

### P0-1: Fix .gitignore & stop tracking binaries
- [x] Update .gitignore with comprehensive patterns
- [x] Add `*.ckpt`, `*.pkl` patterns (except scaler.pkl)
- [x] Run `git rm --cached -r checkpoints/ lightning_logs/`
- [x] Run `git rm --cached models/*.ckpt models/*.pkl` (keep scaler.pkl and CSV)
- [x] Commit: "chore: ignore env/logs/checkpoints and stop tracking binaries"

### P0-2: Notebook hygiene
- [x] Add .pre-commit-config.yaml with nbstripout
- [x] Add .gitattributes for notebook filter
- [x] Run nbstripout on existing notebooks (optional, strips on commit)
- [x] Commit: "chore: strip notebook outputs with nbstripout + pre-commit"

### P0-3: Environment consolidation
- [x] Update environment.yml with missing dependencies (torch, lightning, etc.)
- [x] Add comment to requirements.txt marking it as derived
- [x] Add script to export requirements from conda
- [x] Commit: "chore: document env as source of truth; add requirements export"

## P1 (Week 1) - Single Source of Truth

### P1-1: Move FFT logic to src/features.py
- [x] Create src/features.py with functions from notebook 03
- [x] Update notebooks to import from src.features
- [x] Commit: "refactor: move FFT feature extraction to src/features"

### P1-2: Add smoke tests
- [x] Create tests/test_imports.py
- [x] Create tests/test_smoke.py (model instantiation + FFT)
- [x] Add `make test` target to Makefile
- [x] Commit: "test: add smoke tests for features and model"

### P1-3: Inference script
- [x] Create scripts/predict.py CLI
- [x] Create docs/INFERENCE_GUIDE.md
- [x] Commit: "feat: add CLI inference script"

## P2 (Future) - Not Implemented

- [ ] DVC for data versioning
- [ ] MLflow for experiment tracking
- [ ] CI/CD pipeline
- [ ] Docker containerization

## Verification Commands

```bash
# P0 verification
git status  # Should show only modernization changes

# P1 verification
pytest -q   # Should pass in < 5s
python scripts/predict.py --help  # Should show usage

# Full pipeline (optional)
make run-notebooks  # Should complete without errors
```
