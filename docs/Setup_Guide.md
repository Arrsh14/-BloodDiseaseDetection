# Setup Guide

## Requirements
- Python 3.x, macOS (tested on Apple Silicon / M2)
- Homebrew (for `libomp`, required by XGBoost on macOS)

## 1. Clone and set up environment

```bash
cd BloodDiseaseDetection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Install system dependency for XGBoost (macOS)

XGBoost requires the OpenMP runtime, which isn't installed by default on macOS:

```bash
brew install libomp
```

## 3. Download image datasets

- **Leukemia**: download the C-NMC 2019 dataset from Kaggle, place into
  `data/raw/leukemia/all/` (cancer) and `data/raw/leukemia/hem/` (healthy)
- **Malaria**: download the NIH Malaria dataset from Kaggle, place into
  `data/raw/malaria/Parasitized/` and `data/raw/malaria/Uninfected/`

## 4. Generate the synthetic tabular dataset

```bash
python3 -m src.data.generate_tabular_data
python3 -m src.data.preprocess_tabular
```

## 5. Train all three models

```bash
python3 -m src.training.train_leukemia_cnn
python3 -m src.training.train_malaria_cnn
python3 -m src.training.train_tabular_model
```

Trained models are saved to `saved_models/`.

## 6. (Optional) Run evaluation and Grad-CAM

```bash
python3 -m src.evaluation.evaluate_leukemia
python3 -m src.evaluation.evaluate_malaria
python3 -m src.explainability.gradcam_leukemia
```

## 7. Run inference from the command line

```bash
python3 -m src.inference.predict --image path/to/some/image.png
```

## 8. Run the demo app

```bash
streamlit run app/streamlit_app.py
```

Opens at `http://localhost:8501`.

## Known issues

- **XGBoost + PyTorch MPS conflict**: loading XGBoost and PyTorch in the same
  process can hang on macOS. The inference pipeline works around this by
  running the tabular model in a separate subprocess
  (`src/inference/_tabular_predict_subprocess.py`). If modifying
  `predict.py`, do not import `joblib`/`xgboost` directly in that file.

- **Lab value units**: WBC and platelet counts use ×10³ cells/µL (e.g., a
  normal WBC count is entered as `7.5`, not `7500`). See Dataset_Sources.md
  for the full unit convention.