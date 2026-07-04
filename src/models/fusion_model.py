"""
Fusion logic for the blood disease detection system.

DESIGN DECISION (documented honestly, see docs/Architecture.md for full reasoning):
    The tabular model is the ONLY component in this system trained on all four
    classes (normal / leukemia / malaria / both) jointly. The leukemia and malaria
    CNNs were each trained on separate, disease-specific datasets with no
    overlapping patients and no joint ground truth — the leukemia CNN never saw
    a malaria-infected cell during training, and vice versa.

    Therefore: the tabular model's prediction drives the final diagnosis,
    INCLUDING the "both" case. The relevant CNN (leukemia or malaria) is used
    to provide a supporting confidence score alongside the tabular verdict —
    it confirms/contextualizes the diagnosis using image evidence, but does
    not independently override the tabular model's class decision.

    This is a deliberate choice to avoid overclaiming: we do NOT ask either CNN
    to render an opinion on a condition it was never trained to recognize.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

LABEL_NAMES = ["normal", "leukemia", "malaria", "both"]


def fuse_prediction(tabular_probs, leukemia_cnn_prob=None, malaria_cnn_prob=None):
    """
    Combines model outputs into one final diagnosis + explanation.

    Args:
        tabular_probs: array-like of length 4, probabilities for
                       [normal, leukemia, malaria, both] from the XGBoost model.
        leukemia_cnn_prob: float or None. P(cancer) from the leukemia CNN,
                            if a blood smear image was provided.
        malaria_cnn_prob: float or None. P(parasitized) from the malaria CNN,
                           if a blood smear image was provided.

    Returns:
        dict with:
            - "diagnosis": final class name (str)
            - "tabular_confidence": probability of the winning class per tabular model
            - "image_support": dict describing what the relevant CNN(s) reported,
                                 or a note explaining why no image evidence was used
            - "explanation": human-readable summary of how the decision was reached
    """
    tabular_pred_idx = int(max(range(4), key=lambda i: tabular_probs[i]))
    diagnosis = LABEL_NAMES[tabular_pred_idx]
    tabular_confidence = float(tabular_probs[tabular_pred_idx])

    image_support = {}
    explanation_parts = [
        f"Tabular model predicted '{diagnosis}' with {tabular_confidence:.1%} confidence."
    ]

    if diagnosis == "leukemia":
        if leukemia_cnn_prob is not None:
            image_support["leukemia_cnn_p_cancer"] = float(leukemia_cnn_prob)
            explanation_parts.append(
                f"Leukemia CNN independently gave this image a {leukemia_cnn_prob:.1%} "
                f"probability of being cancerous, supporting this diagnosis."
            )
        else:
            explanation_parts.append("No blood smear image was provided to confirm with imaging.")

    elif diagnosis == "malaria":
        if malaria_cnn_prob is not None:
            image_support["malaria_cnn_p_parasitized"] = float(malaria_cnn_prob)
            explanation_parts.append(
                f"Malaria CNN independently gave this image a {malaria_cnn_prob:.1%} "
                f"probability of being parasitized, supporting this diagnosis."
            )
        else:
            explanation_parts.append("No blood smear image was provided to confirm with imaging.")

    elif diagnosis == "both":
        # Both CNNs are relevant here, if an image was provided. Note: since a single
        # smear image typically reflects ONE condition's visual signature, image
        # evidence for "both" is inherently limited — this is a known, disclosed
        # limitation rather than something the system claims to solve.
        if leukemia_cnn_prob is not None:
            image_support["leukemia_cnn_p_cancer"] = float(leukemia_cnn_prob)
        if malaria_cnn_prob is not None:
            image_support["malaria_cnn_p_parasitized"] = float(malaria_cnn_prob)
        if image_support:
            explanation_parts.append(
                "Note: 'both' diagnoses are driven by lab values (the only data source "
                "with joint ground truth for co-occurring conditions). Image evidence is "
                "shown for reference but a single smear image cannot reliably confirm two "
                "simultaneous conditions."
            )
        else:
            explanation_parts.append("No blood smear image was provided to confirm with imaging.")

    else:  # normal
        explanation_parts.append("No signs of leukemia or malaria detected in lab values.")

    return {
        "diagnosis": diagnosis,
        "tabular_confidence": tabular_confidence,
        "image_support": image_support,
        "explanation": " ".join(explanation_parts),
    }


if __name__ == "__main__":
    # Quick sanity-check examples — no real model calls, just testing the logic.
    print("Example 1: tabular predicts leukemia, image confirms")
    result = fuse_prediction(
        tabular_probs=[0.05, 0.80, 0.05, 0.10],
        leukemia_cnn_prob=0.91,
        malaria_cnn_prob=None,
    )
    print(result, "\n")

    print("Example 2: tabular predicts both, both CNN scores available")
    result = fuse_prediction(
        tabular_probs=[0.02, 0.10, 0.08, 0.80],
        leukemia_cnn_prob=0.65,
        malaria_cnn_prob=0.70,
    )
    print(result, "\n")

    print("Example 3: tabular predicts normal, no image provided")
    result = fuse_prediction(
        tabular_probs=[0.85, 0.05, 0.05, 0.05],
    )
    print(result)