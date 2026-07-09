"""
SHAP explainability for the tabular XGBoost model.

Computes SHAP values on the held-out test set to show which lab values
drove which predictions — the tabular equivalent of Grad-CAM for the CNNs.
Produces:
    1. A summary plot: which features matter most overall, across all predictions
    2. Individual force plots: for a few example patients, exactly which
       features pushed the prediction toward/away from the predicted class
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import PROCESSED_DIR, SAVED_MODELS_DIR, ROOT_DIR

FEATURE_COLS = ["wbc_count", "hemoglobin", "platelet_count", "rbc_count", "parasitemia_pct"]
LABEL_NAMES = ["normal", "leukemia", "malaria", "both"]


def main():
    # --- Load trained tabular model and test data ---
    model = joblib.load(SAVED_MODELS_DIR / "tabular_model.pkl")
    test_df = pd.read_csv(PROCESSED_DIR / "tabular" / "test.csv")
    X_test = test_df[FEATURE_COLS]
    y_test = test_df["label"]

    print(f"Loaded test set: {len(X_test)} samples")

    # --- Compute SHAP values ---
    # TreeExplainer is the fast, exact method for tree-based models like XGBoost
    print("Computing SHAP values (this may take a moment)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # For multi-class XGBoost, shap_values is a list of arrays (one per class)
    # or a single 3D array depending on SHAP/XGBoost version — handle both.
    if isinstance(shap_values, list):
        shap_values_per_class = shap_values  # list of (n_samples, n_features) arrays
    else:
        # shape (n_samples, n_features, n_classes) -> split into list per class
        shap_values_per_class = [shap_values[:, :, i] for i in range(len(LABEL_NAMES))]

    out_dir = ROOT_DIR / "results" / "shap_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Summary plot: overall feature importance per class ---
    for class_idx, class_name in enumerate(LABEL_NAMES):
        plt.figure()
        shap.summary_plot(
            shap_values_per_class[class_idx],
            X_test,
            feature_names=FEATURE_COLS,
            show=False,
        )
        plt.title(f"SHAP Summary — driving factors for '{class_name}' predictions")
        plt.tight_layout()
        save_path = out_dir / f"shap_summary_{class_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

    # --- Individual example explanations: one correctly-classified case per class ---
    predictions = model.predict(X_test)
    for class_idx, class_name in enumerate(LABEL_NAMES):
        # find a test sample that's correctly predicted as this class
        matches = np.where((y_test.values == class_idx) & (predictions == class_idx))[0]
        if len(matches) == 0:
            print(f"No correctly-classified example found for class '{class_name}', skipping")
            continue

        sample_idx = matches[0]
        sample_row = X_test.iloc[sample_idx]

        print(f"\nExample — '{class_name}' prediction (test row {sample_idx}):")
        print(sample_row.to_dict())

        plt.figure()
        shap.bar_plot(
            shap_values_per_class[class_idx][sample_idx],
            feature_names=FEATURE_COLS,
            show=False,
        )
        plt.title(f"SHAP feature contributions — example '{class_name}' prediction")
        plt.tight_layout()
        save_path = out_dir / f"shap_example_{class_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

    print(f"\nAll SHAP plots saved to: {out_dir}")
    print("\nInterpretation guide:")
    print("- Summary plots show which features matter most overall for each class")
    print("- Example plots show exactly which features drove ONE specific prediction")
    print("- Positive SHAP value = pushed prediction TOWARD that class")
    print("- Negative SHAP value = pushed prediction AWAY from that class")


if __name__ == "__main__":
    main()