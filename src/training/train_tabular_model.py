"""
Training script for the tabular XGBoost model.
Loads preprocessed train/val/test CSVs, trains, evaluates, and saves the model.
"""

import pandas as pd
import joblib
import json

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.models.tabular_model import build_tabular_model
from src.utils.config import PROCESSED_DIR, SAVED_MODELS_DIR

FEATURE_COLS = ["wbc_count", "hemoglobin", "platelet_count", "rbc_count", "parasitemia_pct"]
TARGET_COL = "label"

TABULAR_PROCESSED_DIR = PROCESSED_DIR / "tabular"


def load_split(filename):
    df = pd.read_csv(TABULAR_PROCESSED_DIR / filename)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y


def main():
    print("Loading preprocessed splits...")
    X_train, y_train = load_split("train.csv")
    X_val, y_val = load_split("val.csv")
    X_test, y_test = load_split("test.csv")

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Load label map for readable class names in the report
    with open(TABULAR_PROCESSED_DIR / "label_map.json") as f:
        label_map = json.load(f)  # e.g. {"normal": 0, "leukemia": 1, ...}
    class_names = [name for name, _ in sorted(label_map.items(), key=lambda x: x[1])]

    model = build_tabular_model(num_classes=len(class_names))

    print("\nTraining XGBoost model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True,
    )

    # Evaluate on validation set
    val_preds = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"\nValidation accuracy: {val_acc:.4f}")
    print("\nValidation classification report:")
    print(classification_report(y_val, val_preds, target_names=class_names))

    # Save model
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = SAVED_MODELS_DIR / "tabular_model.pkl"
    joblib.dump(model, save_path)
    print(f"Model saved to: {save_path}")

    # Final test set evaluation
    test_preds = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print("\nTest classification report:")
    print(classification_report(y_test, test_preds, target_names=class_names))
    print("\nTest confusion matrix:")
    print(confusion_matrix(y_test, test_preds))


if __name__ == "__main__":
    main()