# 🩸 Blood Disease Detection

A multi-modal diagnostic system that predicts **leukemia**, **malaria**, **both**, or **normal** from a blood smear image and CBC (Complete Blood Count) lab values — combining two CNNs, an XGBoost tabular model, and a fusion layer with Grad-CAM explainability.

Built as a portfolio project to demonstrate end-to-end ML engineering: data sourcing, transfer learning, model evaluation, explainability, multi-model fusion, and a working interactive demo.

> ⚠️ **Not a diagnostic tool.** This is a research/portfolio project. The tabular model is trained on synthetic data (see [Dataset Sources](docs/Dataset_Sources.md)).

---

## What it does

Upload a blood smear image + enter CBC lab values → get a fused diagnosis with supporting evidence from both the image and the lab values, plus a Grad-CAM heatmap showing where the relevant CNN focused.

## Architecture

Three independently-trained models feed into a fusion layer:

| Component | Model | Task | Result |
|---|---|---|---|
| Leukemia CNN | ResNet18 (transfer learning) | Cancer vs healthy | 85.4% accuracy, AUC 0.91 |
| Malaria CNN | ResNet18 (transfer learning) | Parasitized vs uninfected | 95.5% accuracy, AUC 0.99 |
| Tabular model | XGBoost | 4-class: normal/leukemia/malaria/both | Trained on synthetic CBC data |
| Fusion | Rule-based | Combines tabular verdict + relevant CNN confidence | — |

Full design rationale, including two real engineering issues found and fixed along the way (an XGBoost/PyTorch process conflict, and a Grad-CAM attention finding), is documented in [`docs/Architecture.md`](docs/Architecture.md).

## Demo

```bash
streamlit run app/streamlit_app.py
```

Upload a blood smear image, enter CBC values, and get a diagnosis with a full probability breakdown, both CNN confidences, and a Grad-CAM visualization.

## Project structure

BloodDiseaseDetection/
├── data/              # raw + processed datasets (images + synthetic tabular CBC data)
├── src/
│   ├── data/          # dataset classes, preprocessing, synthetic data generation
│   ├── models/        # CNN and XGBoost model definitions
│   ├── training/       # training scripts for all 3 models
│   ├── evaluation/     # metrics, confusion matrices, classification reports
│   ├── explainability/ # Grad-CAM
│   ├── inference/      # end-to-end prediction pipeline + fusion logic
│   └── utils/          # config, device handling
├── app/               # Streamlit demo
├── saved_models/      # trained model weights
├── results/           # evaluation outputs, Grad-CAM images
└── docs/              # architecture, dataset sources, setup guide

## Key results

**Leukemia CNN** — Test accuracy 85.38% | Sensitivity 88.54% | Specificity 78.59% | AUC 0.9078
**Malaria CNN** — Test accuracy 95.48% | Sensitivity 94.87% | Specificity 96.08% | AUC 0.9909

Both were evaluated beyond raw accuracy (confusion matrix, sensitivity/specificity) since accuracy alone is misleading for imbalanced/medical classification tasks — see [`docs/Architecture.md`](docs/Architecture.md) for the full reasoning.

## Setup

Full instructions in [`docs/Setup_Guide.md`](docs/Setup_Guide.md). Quick version:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install libomp   # required by XGBoost on macOS

# generate synthetic tabular data
python3 -m src.data.generate_tabular_data
python3 -m src.data.preprocess_tabular

# train all 3 models
python3 -m src.training.train_leukemia_cnn
python3 -m src.training.train_malaria_cnn
python3 -m src.training.train_tabular_model

# run the demo
streamlit run app/streamlit_app.py
```

## Known limitations

- **Tabular data is synthetic**, grounded in published clinical CBC reference ranges rather than real patient records (no combined public dataset exists — see [`docs/Dataset_Sources.md`](docs/Dataset_Sources.md)).
- **Both image and lab values are required** — the system doesn't support partial (image-only or lab-only) input, a deliberate scope decision explained in [`docs/Architecture.md`](docs/Architecture.md).
- **Grad-CAM attention shows some edge-bias** — model attention on the leukemia CNN skews toward cell boundaries rather than purely interior structure. Investigated and documented, not fully resolved — see [`docs/Architecture.md`](docs/Architecture.md).

## Docs

- [`docs/Architecture.md`](docs/Architecture.md) — system design, fusion logic, engineering findings
- [`docs/Dataset_Sources.md`](docs/Dataset_Sources.md) — dataset citations, synthetic data methodology
- [`docs/Setup_Guide.md`](docs/Setup_Guide.md) — full setup and run instructions

## Tech stack

PyTorch, torchvision (ResNet18), XGBoost, scikit-learn, pytorch-grad-cam, Streamlit, pandas, NumPy