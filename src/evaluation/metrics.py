"""
Reusable evaluation metrics for binary classification models.
Used by evaluate_leukemia.py and evaluate_malaria.py.
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


def get_predictions(model, loader, device):
    """
    Runs the model on a full DataLoader and collects predictions,
    true labels, and predicted probabilities (for AUC).
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []  # probability of the positive class (class 1)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # prob of class 1
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_confusion_matrix(y_true, y_pred):
    """
    Returns confusion matrix as [[TN, FP], [FN, TP]].
    Label convention: 0 = negative class (healthy/uninfected), 1 = positive class (cancer/parasitized)
    """
    return confusion_matrix(y_true, y_pred)


def compute_sensitivity_specificity(y_true, y_pred):
    """
    Sensitivity (recall) = TP / (TP + FN)  -> % of actual positives correctly caught
    Specificity = TN / (TN + FP)  -> % of actual negatives correctly caught
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return sensitivity, specificity


def compute_auc(y_true, y_probs):
    """
    AUC (Area Under ROC Curve) — a single number summarizing how well the
    model separates the two classes across all possible thresholds, not just
    at the default 0.5 cutoff. 1.0 = perfect, 0.5 = random guessing.
    """
    return roc_auc_score(y_true, y_probs)


def print_full_report(y_true, y_pred, y_probs, class_names=("Negative", "Positive")):
    """
    Prints confusion matrix, sensitivity, specificity, and AUC in a readable format.
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    sensitivity, specificity = compute_sensitivity_specificity(y_true, y_pred)
    auc = compute_auc(y_true, y_probs)
    accuracy = (y_true == y_pred).mean()

    print("=" * 50)
    print("Confusion Matrix")
    print(f"                Predicted {class_names[0]}   Predicted {class_names[1]}")
    print(f"Actual {class_names[0]:<10}  {cm[0][0]:<20} {cm[0][1]}")
    print(f"Actual {class_names[1]:<10}  {cm[1][0]:<20} {cm[1][1]}")
    print("-" * 50)
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}  (% of actual {class_names[1]} correctly caught)")
    print(f"Specificity: {specificity:.4f}  (% of actual {class_names[0]} correctly caught)")
    print(f"AUC:         {auc:.4f}")
    print("=" * 50)