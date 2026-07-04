"""
Evaluates the trained leukemia CNN on the held-out test set.

Uses the EXACT same test indices saved during training (leukemia_test_indices.npy)
so there is no leakage or accidental re-splitting.

Reports:
    - Accuracy
    - Sensitivity (recall for cancer/positive class = 1)
    - Specificity (recall for healthy/negative class = 0)
    - AUC (using predicted probabilities, not just hard labels)
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
from src.data.dataset_leukemia import LeukemiaDataset
from src.data.preprocess_images import get_eval_transforms
from src.models.cnn_leukemia import build_leukemia_model
from src.utils.config import LEUKEMIA_RAW, SAVED_MODELS_DIR, RESULTS_DIR, BATCH_SIZE
from src.utils.device import get_device


def main():
    device = get_device()
    print(f"Using device: {device}")

    # --- Load the exact test set used during training ---
    test_idx = np.load(SAVED_MODELS_DIR / "leukemia_test_indices.npy")
    print(f"Test set size: {len(test_idx)}")

    eval_dataset_full = LeukemiaDataset(root_dir=LEUKEMIA_RAW, transform=get_eval_transforms())
    test_ds = Subset(eval_dataset_full, test_idx)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- Load the trained model ---
    model = build_leukemia_model(unfreeze_layer4=True).to(device)
    model.load_state_dict(torch.load(SAVED_MODELS_DIR / "leukemia_cnn_final.pth", map_location=device))
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []  # probability of class 1 (cancer), needed for AUC

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)[:, 1]  # prob of "cancer" class
            preds = outputs.argmax(dim=1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # --- Core metrics ---
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn)  # recall for cancer (positive class)
    specificity = tn / (tn + fp)  # recall for healthy (negative class)
    auc = roc_auc_score(all_labels, all_probs)

    print(f"\nTest Accuracy:    {acc:.4f}")
    print(f"Sensitivity:      {sensitivity:.4f}  (catching actual cancer cases)")
    print(f"Specificity:      {specificity:.4f}  (correctly clearing healthy cases)")
    print(f"AUC:              {auc:.4f}")

    print("\nClassification report:")
    print(classification_report(all_labels, all_preds, target_names=["healthy", "cancer"]))

    print("Confusion matrix:")
    print(cm)

    # --- Save confusion matrix plot ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cm_dir = RESULTS_DIR / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["healthy", "cancer"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Leukemia CNN - Confusion Matrix (Test Set)")
    plt.tight_layout()
    out_path = cm_dir / "leukemia_confusion_matrix.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved confusion matrix plot to {out_path}")

    # --- Append results to a metrics summary file ---
    metrics_path = RESULTS_DIR / "metrics_report.md"
    with open(metrics_path, "a") as f:
        f.write("\n## Leukemia CNN (ResNet18, two-phase transfer learning)\n")
        f.write(f"- Test Accuracy: {acc:.4f}\n")
        f.write(f"- Sensitivity: {sensitivity:.4f}\n")
        f.write(f"- Specificity: {specificity:.4f}\n")
        f.write(f"- AUC: {auc:.4f}\n")
        f.write(f"- Confusion Matrix: [[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]]\n")
    print(f"Appended results to {metrics_path}")


if __name__ == "__main__":
    main()