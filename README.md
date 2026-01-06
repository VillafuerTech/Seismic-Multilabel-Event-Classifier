# Project ML2024: Multi-Label Classification of Seismic Signals

## Overview

This repository implements a full end-to-end data science pipeline for multi-label classification of seismic signals captured in `.at2` format. The objective is to preprocess raw seismic data, extract spectral features using FFT, train and evaluate multiple machine learning models, and deploy the optimal model for production use.

## Quick Start

```bash
# 1. Create and activate conda environment
conda env create -f environment.yml
conda activate ml_project

# 2. Run smoke tests to verify installation
make test

# 3. Run inference on sample data (requires trained model)
python scripts/predict.py --input data/sample_input.json
```

## Repository Structure

```
├── data/
│   ├── raw/              # Original `.at2` data files
│   ├── interim/          # Intermediate data after preprocessing
│   ├── processed/        # Final feature matrices and labels
│   └── sample_input.json # Synthetic sample for testing inference
├── notebooks/            # Jupyter Notebooks (01-10)
├── src/
│   ├── features.py       # FFT-based feature extraction
│   └── seismic_model.py  # PyTorch Lightning model definition
├── scripts/
│   ├── predict.py        # CLI inference script
│   └── export_requirements.sh
├── tests/                # Unit tests (smoke tests < 5s)
├── docs/
│   ├── INFERENCE_GUIDE.md
│   └── modernization/    # Modernization plan and changelog
├── models/               # Trained models and scalers (binaries not tracked)
├── environment.yml       # Conda environment (source of truth)
├── requirements.txt      # Pip requirements (derived)
├── Makefile              # Automation targets
└── .pre-commit-config.yaml
```

## Prerequisites

- [Conda](https://docs.conda.io/en/latest/) installed
- Python 3.12 (specified in `environment.yml`)

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/ml2024-seismic-classification.git
   cd ml2024-seismic-classification
   ```

2. Create and activate the Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate ml_project
   ```

3. (Optional) Install pre-commit hooks for notebook hygiene:
   ```bash
   pre-commit install
   ```

## Usage

### Run Tests

```bash
make test
# or directly:
pytest tests/ -v --tb=short -q
```

### Execute All Notebooks

Run the complete pipeline and generate executed notebook outputs:

```bash
make run-notebooks
```

### Run Inference

See [docs/INFERENCE_GUIDE.md](docs/INFERENCE_GUIDE.md) for detailed instructions.

```bash
# Basic usage
python scripts/predict.py --input data/sample_input.json

# With custom threshold
python scripts/predict.py --input data.json --threshold 0.6 --output results.json
```

## Testing

Smoke tests verify core functionalities in under 5 seconds:

```bash
make test
```

Tests include:
- Import verification for all dependencies
- Feature extraction with synthetic sine waves
- Model instantiation and forward pass

## Notebook Descriptions

1. **01_Data_Preprocessing.ipynb**: Data ingestion from `.at2`, cleanup, and normalization.
2. **02_Exploratory_Data_Analysis.ipynb**: Exploratory data analysis and visualization of raw signals and label distributions.
3. **03_Feature_Engineering.ipynb**: FFT computation and construction of feature vectors.
4. **04_Baseline_Model.ipynb**: Implementation and evaluation of a baseline classifier.
5. **05_SVM_Model.ipynb**: Training and evaluation of an SVM with RBF kernel.
6. **06_Random_Forest_Model.ipynb**: Training and evaluation of a Random Forest multi-label classifier.
7. **07_Neural_Network_Model.ipynb**: Implementation of a Feedforward or 1D-CNN model for multi-label classification.
8. **08_Model_Comparison.ipynb**: Systematic comparison of model performance metrics and runtime.
9. **09_Results_Visualization.ipynb**: In-depth analysis of best model results, error cases, and learning curves.
10. **10_Final_Report_and_Deployment.ipynb**: Executive summary, model export, and deployment guidelines.

## Artifacts

- **Model artifacts**: Serialized final model (`model.joblib`) or TensorFlow `saved_model/` directory.
- **Visualizations**: Performance plots and comparison charts.
- **Reports**: Optional HTML/PDF export of report notebooks via `nbconvert`.

## Contributing

Contributions are welcome. Please open issues for bug reports or feature requests, and submit pull requests for enhancements.

## License

This project is intended for academic purposes and is not licensed for commercial use.
