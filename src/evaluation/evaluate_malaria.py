"""
Evaluates the trained malaria CNN on the held-out test set.

Uses the EXACT same test indices saved during training (malaria_test_indices.npy).

Reports:
    - Accuracy
    - Sensitivity (recall for parasitized/positive class = 1)
    - Specificity (recall for uninfected/negative class = 0)
    - AUC
    - Confusion matrix (saved as a plot to results/confusion_matrices/)
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.data.dataset_malaria import MalariaDataset
from src.data.preprocess_images import get_eval_transforms
from src.models.cnn_malaria import build_malaria_model
from src.utils.config import MALARIA_RAW, SAVED_MODELS_DIR, RESULTS_DIR, BATCH_SIZE
from src.utils.device import get_device


def main():
    device = get_device()
    print(f"Using device: {device}")

    test_idx = np.load(SAVED_MODELS_DIR / "malaria_test_indices.npy")
    print(f"Test set size: {len(test_idx)}")

    eval_dataset_full = MalariaDataset(root_dir=MALARIA_RAW, transform=get_eval_transforms())
    test_ds = Subset(eval_dataset_full, test_idx)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_malaria_model(unfreeze_layer4=True).to(device)
    model.load_state_dict(torch.load(SAVED_MODELS_DIR / "malaria_cnn_final.pth", map_location=device))
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)[:, 1]  # prob of "parasitized"
            preds = outputs.argmax(dim=1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    auc = roc_auc_score(all_labels, all_probs)

    print(f"\nTest Accuracy:    {acc:.4f}")
    print(f"Sensitivity:      {sensitivity:.4f}  (catching actual parasitized cases)")
    print(f"Specificity:      {specificity:.4f}  (correctly clearing uninfected cases)")
    print(f"AUC:              {auc:.4f}")

    print("\nClassification report:")
    print(classification_report(all_labels, all_preds, target_names=["uninfected", "parasitized"]))

    print("Confusion matrix:")
    print(cm)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cm_dir = RESULTS_DIR / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["uninfected", "parasitized"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Malaria CNN - Confusion Matrix (Test Set)")
    plt.tight_layout()
    out_path = cm_dir / "malaria_confusion_matrix.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved confusion matrix plot to {out_path}")

    metrics_path = RESULTS_DIR / "metrics_report.md"
    with open(metrics_path, "a") as f:
        f.write("\n## Malaria CNN (ResNet18, two-phase transfer learning)\n")
        f.write(f"- Test Accuracy: {acc:.4f}\n")
        f.write(f"- Sensitivity: {sensitivity:.4f}\n")
        f.write(f"- Specificity: {specificity:.4f}\n")
        f.write(f"- AUC: {auc:.4f}\n")
        f.write(f"- Confusion Matrix: [[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]]\n")
    print(f"Appended results to {metrics_path}")


if __name__ == "__main__":
    main()