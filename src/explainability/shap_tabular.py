"""
SHAP explainability for the tabular XGBoost model (leukemia / malaria / both / normal).

Generates two things, saved to results/:
    1. shap_summary_<class>.png  - global feature importance (beeswarm) per class
    2. shap_waterfall_example.png - explanation for one single example prediction

Uses the model saved by train_tabular_model.py and the held-out test set,
so explanations are shown on data the model did not train on.
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import PROCESSED_DIR, SAVED_MODELS_DIR, RESULTS_DIR

FEATURE_COLS = ["wbc_count", "hemoglobin", "platelet_count", "rbc_count", "parasitemia_pct"]
TABULAR_PROCESSED_DIR = PROCESSED_DIR / "tabular"


def main():
    # Load model
    model = joblib.load(SAVED_MODELS_DIR / "tabular_model.pkl")

    # Load label map (int -> class name) for readable plot titles
    with open(TABULAR_PROCESSED_DIR / "label_map.json") as f:
        label_map = json.load(f)  # e.g. {"normal": 0, "leukemia": 1, ...}
    inv_label_map = {v: k for k, v in label_map.items()}

    # Load test set
    test_df = pd.read_csv(TABULAR_PROCESSED_DIR / "test.csv")
    X_test = test_df[FEATURE_COLS]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Global explanation: SHAP summary (beeswarm) plot per class ---
    print("Computing SHAP values for test set...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)  # shape: (n_samples, n_features, n_classes) for multiclass

    n_classes = shap_values.values.shape[2]
    for class_idx in range(n_classes):
        class_name = inv_label_map[class_idx]
        plt.figure()
        shap.summary_plot(
            shap_values.values[:, :, class_idx],
            X_test,
            show=False,
        )
        plt.title(f"SHAP Summary — class: {class_name}")
        plt.tight_layout()
        out_path = RESULTS_DIR / f"shap_summary_{class_name}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")

    # --- 2. Local explanation: waterfall plot for one example ---
    # Pick one test-set example to explain in detail — first row, but you can
    # change `example_idx` to inspect any specific patient.
    example_idx = 0
    example_row = X_test.iloc[example_idx]
    true_label = inv_label_map[test_df.iloc[example_idx]["label"]]
    predicted_label_idx = model.predict(X_test.iloc[[example_idx]])[0]
    predicted_label = inv_label_map[predicted_label_idx]

    print(f"\nExample row (index {example_idx}):")
    print(example_row)
    print(f"True label: {true_label} | Predicted label: {predicted_label}")

    # Explain the SHAP values for the predicted class specifically
    plt.figure()
    shap.plots.waterfall(shap_values[example_idx, :, predicted_label_idx], show=False)
    plt.title(f"SHAP Waterfall — true: {true_label}, predicted: {predicted_label}")
    plt.tight_layout()
    out_path = RESULTS_DIR / "shap_waterfall_example.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()