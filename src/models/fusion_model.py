"""
Fusion logic for the blood disease detection system — LEARNED FUSION MODEL.

DESIGN HISTORY (see docs/Architecture.md for full reasoning):
    An earlier version of this module used rule-based fusion: the tabular
    model's prediction drove the diagnosis outright, with the relevant CNN's
    confidence shown as supporting evidence but never weighed against the
    tabular verdict. That approach was simple but not a genuine "fusion" —
    it never let CNN evidence override or adjust an uncertain tabular call.

    This version uses a trained logistic regression meta-model
    (src/training/train_fusion_model.py) that takes all 6 signals —
    [p_normal, p_leukemia, p_malaria, p_both] from the tabular model, plus
    leukemia_cnn_prob and malaria_cnn_prob — and learns how to combine them,
    including cases where the tabular signal is ambiguous/borderline (see
    generate_fusion_training_data.py for how ambiguous training cases were
    constructed) and the CNN evidence should carry more weight.
"""

import sys
import json
import pandas as pd
from pathlib import Path


import joblib
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import SAVED_MODELS_DIR

LABEL_NAMES = ["normal", "leukemia", "malaria", "both"]

_fusion_model = joblib.load(SAVED_MODELS_DIR / "fusion_model.pkl")
with open(SAVED_MODELS_DIR / "fusion_feature_order.json") as f:
    _FEATURE_ORDER = json.load(f)  # e.g. [normal, leukemia, malaria, both, leukemia_cnn_prob, malaria_cnn_prob]


def fuse_prediction(tabular_probs, leukemia_cnn_prob=None, malaria_cnn_prob=None):
    """
    Combines model outputs into one final diagnosis + explanation, using the
    trained fusion meta-model rather than a fixed rule.

    Args:
        tabular_probs: array-like of length 4, probabilities for
                       [normal, leukemia, malaria, both] from the XGBoost model.
        leukemia_cnn_prob: float or None. P(cancer) from the leukemia CNN.
        malaria_cnn_prob: float or None. P(parasitized) from the malaria CNN.
            NOTE: the fusion model was trained assuming both CNN scores are
            always present (this system requires image + lab values jointly —
            see docs/Architecture.md). If either is missing, it's filled with
            0.0 as a neutral placeholder; this is a known edge case, not a
            fully-supported partial-input path.

    Returns:
        dict with:
            - "diagnosis": final class name (str), chosen by the fusion model
            - "tabular_confidence": probability of the tabular model's own top class
                                     (kept for transparency/comparison, not what
                                     drives the final diagnosis anymore)
            - "fusion_confidence": the fusion model's confidence in its own decision
            - "image_support": dict of the CNN scores that were fed into fusion
            - "explanation": human-readable summary of how the decision was reached
    """
    tabular_probs = list(tabular_probs)
    tabular_pred_idx = int(max(range(4), key=lambda i: tabular_probs[i]))
    tabular_top_class = LABEL_NAMES[tabular_pred_idx]
    tabular_confidence = float(tabular_probs[tabular_pred_idx])

    leuk_prob = float(leukemia_cnn_prob) if leukemia_cnn_prob is not None else 0.0
    mal_prob = float(malaria_cnn_prob) if malaria_cnn_prob is not None else 0.0

    feature_values = {
        "normal": tabular_probs[0],
        "leukemia": tabular_probs[1],
        "malaria": tabular_probs[2],
        "both": tabular_probs[3],
        "leukemia_cnn_prob": leuk_prob,
        "malaria_cnn_prob": mal_prob,
    }
    X = pd.DataFrame([[feature_values[col] for col in _FEATURE_ORDER]], columns=_FEATURE_ORDER)

    fusion_pred = _fusion_model.predict(X)[0]
    fusion_probs = _fusion_model.predict_proba(X)[0]
    fusion_classes = list(_fusion_model.classes_)
    fusion_confidence = float(fusion_probs[fusion_classes.index(fusion_pred)])

    diagnosis = fusion_pred

    image_support = {}
    if leukemia_cnn_prob is not None:
        image_support["leukemia_cnn_p_cancer"] = leuk_prob
    if malaria_cnn_prob is not None:
        image_support["malaria_cnn_p_parasitized"] = mal_prob

    explanation_parts = [
        f"Fusion model predicted '{diagnosis}' with {fusion_confidence:.1%} confidence, "
        f"combining tabular lab values (top class: '{tabular_top_class}' at "
        f"{tabular_confidence:.1%}) with CNN image evidence."
    ]

    if diagnosis != tabular_top_class:
        explanation_parts.append(
            f"Note: the fusion model's diagnosis differs from the tabular model's "
            f"standalone top prediction ('{tabular_top_class}') — image evidence "
            f"shifted the final decision."
        )

    if diagnosis == "both":
        explanation_parts.append(
            "Note: a single smear image typically reflects one condition's visual "
            "signature, so image evidence for 'both' has inherent limits — shown "
            "here for reference, weighed by the fusion model alongside lab values."
        )

    return {
        "diagnosis": diagnosis,
        "tabular_confidence": tabular_confidence,
        "fusion_confidence": fusion_confidence,
        "image_support": image_support,
        "explanation": " ".join(explanation_parts),
    }


if __name__ == "__main__":
    print("Example 1: tabular predicts leukemia, image confirms")
    result = fuse_prediction(
        tabular_probs=[0.05, 0.80, 0.05, 0.10],
        leukemia_cnn_prob=0.91,
        malaria_cnn_prob=0.10,
    )
    print(result, "\n")

    print("Example 2: tabular uncertain, CNN evidence may shift the call")
    result = fuse_prediction(
        tabular_probs=[0.30, 0.35, 0.25, 0.10],
        leukemia_cnn_prob=0.20,
        malaria_cnn_prob=0.85,
    )
    print(result)