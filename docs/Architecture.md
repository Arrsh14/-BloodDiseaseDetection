# Architecture

## Overview

This system predicts leukemia, malaria, both, or normal from two inputs:
a blood smear image and a set of CBC (Complete Blood Count) lab values.
It combines three independently-trained models through a fusion layer.

## Components

1. **Leukemia CNN** — ResNet18 (transfer learning), binary classifier
   (cancer vs healthy), trained on the C-NMC 2019 dataset.
   Test accuracy: 85.38% | Sensitivity: 88.54% | Specificity: 78.59% | AUC: 0.91

2. **Malaria CNN** — ResNet18 (transfer learning), binary classifier
   (parasitized vs uninfected), trained on the NIH Malaria dataset.
   Test accuracy: 95.48% | Sensitivity: 94.87% | Specificity: 96.08% | AUC: 0.99

3. **Tabular model** — XGBoost multi-class classifier (4 classes: normal,
   leukemia, malaria, both), trained on a synthetic CBC dataset grounded in
   published clinical reference ranges (see Dataset_Sources.md).

4. **Fusion layer** — the tabular model's prediction determines the final
   diagnosis. The CNN corresponding to that diagnosis (leukemia or malaria)
   is used as supporting visual confidence. Both CNNs always run regardless
   of the tabular verdict, so their raw outputs are available for inspection.

## Design decision: both image and lab values are required

The system requires both inputs rather than supporting partial input
(image-only or lab-only). This was a deliberate choice:
- Matches the original project goal — a joint image + lab value diagnostic tool.
- The fusion logic fundamentally depends on the tabular model's 4-class output
  to decide the final diagnosis (including "both", which no single CNN can
  express). An image-only path would have no mechanism to reach that verdict.
- Avoids conditional/fallback logic for partial-input edge cases, keeping
  both the inference pipeline and the demo app simpler.

Known limitation: the system cannot function as a pure image-only screening
tool. This is a scope decision, not an oversight.

## Engineering finding: XGBoost + PyTorch (MPS) process conflict

On macOS, loading a pickled XGBoost model and PyTorch (with MPS/Apple Silicon
GPU support) in the same Python process causes unpredictable hangs and
segfaults. This was traced through systematic isolation testing — the freeze
point moved between the pretrained-weight download, the `.to("mps")` device
transfer, and even basic image transforms, depending on what had already
been imported. This pattern (recurring freezes at different points) pointed
to a native library / threading conflict between XGBoost's OpenMP runtime
and PyTorch's MPS backend, rather than a bug in any single piece of code.

**Fix:** the tabular (XGBoost) model runs in a separate subprocess
(`src/inference/_tabular_predict_subprocess.py`), completely isolated from
the PyTorch-based CNN inference in `predict.py`. `predict.py` never imports
`joblib`/`xgboost` directly — it invokes the subprocess and parses its JSON
output. This is a standard, defensible way to handle a cross-library native
conflict of this kind, without touching library internals.

## Finding: Grad-CAM attention and edge-bias

Grad-CAM analysis on the leukemia CNN (comparing `layer3` vs `layer4`
activations) showed model attention consistently skewing toward cell
*boundaries* rather than the cell interior, in both correctly- and
incorrectly-classified examples. Raw dataset images were checked and ruled
out as a source of framing bias (cells are consistently centered on a plain
black background in both classes).

Two explanations remain plausible and were not fully disambiguated:
1. Cell membrane/shape irregularities are genuinely diagnostically relevant,
   so edge-focused attention isn't necessarily wrong.
2. The model may be partly keying off the high-contrast black-background-to-
   cell edge transition — an easy visual signal — rather than purely
   interior morphology (chromatin texture, granularity).

This is documented as an open limitation rather than resolved, since fixing
it conclusively would require architecture changes beyond this project's
scope. The demo app uses `layer4` Grad-CAM (per model consistency); `layer3`
comparisons showed sharper but not fundamentally different attention.

## Data flow

```
Image + Lab Values
       │
       ├──> Tabular model (subprocess) ──> 4-class probabilities
       │
       ├──> Leukemia CNN ──> P(cancer)
       │
       ├──> Malaria CNN ──> P(parasitized)
       │
       └──> Fusion ──> Final diagnosis + explanation + Grad-CAM
```