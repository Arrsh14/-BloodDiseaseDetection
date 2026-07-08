"""
Trains the learned fusion model.

Takes the paired data from generate_fusion_training_data.py (CNN outputs +
lab values + true label), runs each row's lab values through the tabular
model (via subprocess, to avoid XGBoost/PyTorch conflicts), then trains a
logistic regression classifier on the combined 6-feature vector:
    [p_normal, p_leukemia, p_malaria, p_both, leukemia_cnn_prob, malaria_cnn_prob]
    -> true diagnosis

This REPLACES the rule-based fusion logic previously in fusion_model.py.
"""

import sys
import json
import subprocess
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import PROCESSED_DIR, SAVED_MODELS_DIR, RANDOM_SEED

FEATURE_COLS = ["wbc_count", "hemoglobin", "platelet_count", "rbc_count", "parasitemia_pct"]
LABEL_NAMES = ["normal", "leukemia", "malaria", "both"]


def get_tabular_probs(lab_values: dict) -> dict:
    """Calls the isolated tabular subprocess, same as predict.py does."""
    script_path = Path(__file__).resolve().parent.parent / "inference" / "_tabular_predict_subprocess.py"
    lab_values_json = json.dumps(lab_values)

    result = subprocess.run(
        [sys.executable, "-m", "src.inference._tabular_predict_subprocess", lab_values_json],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent.parent,
        timeout=30,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Tabular subprocess failed:\n{result.stderr}")

    return json.loads(result.stdout.strip())


def main():
    input_path = PROCESSED_DIR / "fusion" / "fusion_training_data.csv"
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} paired samples")

    print("Running tabular model on each row (via subprocess)...")
    tabular_prob_rows = []
    for i, row in df.iterrows():
        lab_values = {col: row[col] for col in FEATURE_COLS}
        probs = get_tabular_probs(lab_values)
        tabular_prob_rows.append(probs)
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(df)}")

    tabular_df = pd.DataFrame(tabular_prob_rows)  # columns: normal, leukemia, malaria, both
    tabular_df = tabular_df[LABEL_NAMES]  # enforce consistent column order

    # Build final feature matrix
    X = pd.concat([
        tabular_df.reset_index(drop=True),
        df[["leukemia_cnn_prob", "malaria_cnn_prob"]].reset_index(drop=True),
    ], axis=1)
    y = df["true_label"].reset_index(drop=True)

    print(f"\nFeature matrix shape: {X.shape}")
    print(X.head())

    # Save the full feature+label dataset for inspection/reuse
    fusion_dir = PROCESSED_DIR / "fusion"
    full_data = X.copy()
    full_data["true_label"] = y
    full_data.to_csv(fusion_dir / "fusion_features_full.csv", index=False)

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    # --- Train the fusion model ---
    print("\nTraining logistic regression fusion model...")
    fusion_model = LogisticRegression(max_iter=1000)
    fusion_model.fit(X_train, y_train)

    # --- Evaluate ---
    train_acc = accuracy_score(y_train, fusion_model.predict(X_train))
    test_preds = fusion_model.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)

    print(f"\nTrain accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print("\nTest classification report:")
    print(classification_report(y_test, test_preds))
    print("\nTest confusion matrix (rows=true, cols=pred):")
    print(f"Classes order: {sorted(y.unique())}")
    print(confusion_matrix(y_test, test_preds, labels=sorted(y.unique())))

    # --- Save the trained fusion model ---
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = SAVED_MODELS_DIR / "fusion_model.pkl"
    joblib.dump(fusion_model, save_path)
    print(f"\nFusion model saved to: {save_path}")

    # Save feature column order — predict.py needs this to build the input vector correctly
    feature_order = list(X.columns)
    with open(SAVED_MODELS_DIR / "fusion_feature_order.json", "w") as f:
        json.dump(feature_order, f, indent=2)
    print(f"Feature order saved: {feature_order}")


if __name__ == "__main__":
    main()