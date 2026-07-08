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

## Learned fusion model (replacing rule-based fusion)

An earlier version of this system used rule-based fusion: the tabular
model's top prediction determined the final diagnosis outright, with the
relevant CNN's confidence shown only as supporting evidence — never actually
weighed against or capable of overriding the tabular verdict. This worked,
but wasn't genuine fusion; it was conditional routing.

**Replaced with a trained fusion model.** A logistic regression meta-model
now takes all 6 available signals — the tabular model's 4-class probability
distribution, plus both CNNs' confidence scores — and learns to combine them,
including cases where the tabular signal is ambiguous and CNN evidence
should carry more or less weight.

### Training data challenge and fix

Initial fusion training data was generated using the same distributions the
tabular model was already trained on, so the tabular model solved every
training example perfectly (100% test accuracy on the fusion task) — which
meant the fusion model just learned to copy the tabular model's argmax,
demonstrating nothing about genuine multi-signal combination.

**Fix:** the fusion training data generator (`generate_fusion_training_data.py`)
deliberately blends ~40% of samples' lab values with another class's
distribution (a configurable blend weight), creating genuinely ambiguous
cases while keeping image labels as reliable ground truth. Retraining on this
harder dataset dropped test accuracy from 100% to 88.3% — a meaningful,
credible number reflecting a model that has to actually weigh conflicting
evidence, not one exploiting an artificially easy task. Confusion matrix
analysis confirmed errors were "reasonable" (concentrated between
adjacent/related classes, e.g. malaria↔both, leukemia↔both), not erratic.

### Verified override behavior

Because the trained tabular model turned out to have very sharp,
near-deterministic decision boundaries in practice (manually probing dozens
of lab value combinations consistently returned >97% confidence for one
class), naturally-occurring ambiguous real-world inputs proved difficult to
find. To verify the fusion model's override logic directly, `fuse_prediction()`
was tested with hand-crafted conflicting inputs
(`src/inference/test_fusion_override.py`):

- **Override confirmed**: given an ambiguous tabular signal (normal 42% vs.
  leukemia 38%) paired with a strong leukemia CNN signal (93%), the fusion
  model correctly overrode the tabular model's standalone top pick and
  diagnosed leukemia — explicitly flagging that image evidence shifted the
  decision.
- **Calibrated restraint confirmed**: given a very confident tabular signal
  (leukemia at 97%) paired with a single strongly conflicting CNN score, the
  fusion model correctly stuck with the tabular verdict rather than being
  swayed by one disagreeing source.
- **Honest limitation found**: given an ambiguous tabular signal (normal 45%
  vs. malaria 40%) paired with a very strong malaria CNN signal (95%), the
  fusion model landed on "normal" at only 48% confidence — essentially
  uncertain rather than confidently resolving the conflict either way. This
  is disclosed as a known limitation: the model can produce low-confidence,
  ambiguous outputs under conflicting evidence rather than always resolving
  decisively. This is arguably more honest behavior than false confidence,
  but is not a fully "solved" conflict-resolution mechanism.

  ## End-to-end system evaluation

Beyond evaluating each component in isolation, the complete pipeline
(tabular subprocess + both CNNs + learned fusion model) was evaluated
end-to-end on 120 held-out test cases (30 per class), generated with a
separate random seed from all training data and with the same deliberate
ambiguity-injection technique used in fusion model training (~40% of
samples blend lab values with another class's distribution).

**Result: 87.5% end-to-end accuracy.** Per-class recall: normal 100%,
both 87%, malaria 83%, leukemia 80%. Confusion matrix analysis shows all
misclassifications occur between clinically-related classes (e.g.
malaria↔both, leukemia↔both) — no confusions between unrelated classes
(e.g. normal↔both), indicating the system's errors are concentrated in
genuinely hard/ambiguous cases rather than reflecting broken logic.