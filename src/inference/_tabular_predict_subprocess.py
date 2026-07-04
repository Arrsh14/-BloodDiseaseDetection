"""
Standalone subprocess script for running the tabular XGBoost model.

This is intentionally isolated into its own process, separate from predict.py,
because loading XGBoost and PyTorch (with MPS) in the same Python process
causes unpredictable hangs on macOS (native library / threading conflict
between XGBoost's OpenMP threads and PyTorch's MPS backend). This script
never imports torch — it only ever runs joblib/xgboost — so the conflict
cannot occur here.

Usage (called internally by predict.py via subprocess, not run manually):
    python3 -m src.inference._tabular_predict_subprocess '{"wbc_count": 34.2, ...}'

Prints a JSON result to stdout: {"normal": 0.1, "leukemia": 0.7, ...}
"""

import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import SAVED_MODELS_DIR

import joblib
import numpy as np

FEATURE_COLS = ["wbc_count", "hemoglobin", "platelet_count", "rbc_count", "parasitemia_pct"]
LABEL_NAMES = ["normal", "leukemia", "malaria", "both"]


def main():
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Expected exactly one JSON argument with lab values"}))
        sys.exit(1)

    lab_values = json.loads(sys.argv[1])

    model = joblib.load(SAVED_MODELS_DIR / "tabular_model.pkl")
    row = np.array([[lab_values[col] for col in FEATURE_COLS]])
    probs = model.predict_proba(row)[0]

    result = {name: float(p) for name, p in zip(LABEL_NAMES, probs)}
    print(json.dumps(result))


if __name__ == "__main__":
    main()