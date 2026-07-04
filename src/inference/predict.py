"""
Main inference pipeline for the blood disease detection system.

Requires BOTH a blood smear image and lab values (CBC-style: wbc_count,
hemoglobin, platelet_count, rbc_count, parasitemia_pct) as input — this is
a deliberate design choice (see docs/Architecture.md): the fusion logic
depends on the tabular model's 4-class prediction to determine the final
diagnosis, so lab values are always required; the image is used to run
whichever CNN is relevant and provide supporting visual confidence.

NOTE ON ARCHITECTURE: the tabular (XGBoost) model is run in a separate
subprocess (see _tabular_predict_subprocess.py), not imported directly here.
This is because loading XGBoost and PyTorch/MPS in the same process causes
unpredictable hangs on macOS. This script only ever imports torch — never
joblib/xgboost — so that conflict cannot occur.

Usage:
    from src.inference.predict import predict

    result = predict(
        image_path="path/to/smear.png",
        lab_values={
            "wbc_count": 34.2, "hemoglobin": 8.9,
            "platelet_count": 60.0, "rbc_count": 3.2, "parasitemia_pct": 0.0,
        },
    )
"""

import sys
import json
import subprocess
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn.functional as F
from PIL import Image

from src.data.preprocess_images import get_eval_transforms
from src.models.cnn_leukemia import build_leukemia_model_for_inference
from src.models.cnn_malaria import build_malaria_model_for_inference
from src.models.fusion_model import fuse_prediction, LABEL_NAMES
from src.utils.config import SAVED_MODELS_DIR
from src.utils.device import get_device

FEATURE_COLS = ["wbc_count", "hemoglobin", "platelet_count", "rbc_count", "parasitemia_pct"]

_device = get_device()

_leukemia_model = build_leukemia_model_for_inference().to(_device)
_leukemia_model.load_state_dict(
    torch.load(SAVED_MODELS_DIR / "leukemia_cnn_final.pth", map_location=_device)
)
_leukemia_model.eval()

_malaria_model = build_malaria_model_for_inference().to(_device)
_malaria_model.load_state_dict(
    torch.load(SAVED_MODELS_DIR / "malaria_cnn_final.pth", map_location=_device)
)
_malaria_model.eval()

_eval_transform = get_eval_transforms()


def _get_tabular_probs(lab_values: dict) -> dict:
    """
    Runs the tabular model in a SEPARATE subprocess (never in this process,
    to avoid the XGBoost/PyTorch MPS conflict). Returns a dict of
    {class_name: probability}.
    """
    script_path = Path(__file__).resolve().parent / "_tabular_predict_subprocess.py"
    lab_values_json = json.dumps(lab_values)

    result = subprocess.run(
        [sys.executable, "-m", "src.inference._tabular_predict_subprocess", lab_values_json],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent.parent,  # run from project root
        timeout=30,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Tabular subprocess failed:\n{result.stderr}")

    return json.loads(result.stdout.strip())


def _get_cnn_prob(model, image_tensor) -> float:
    """Runs a binary CNN, returns P(positive class = index 1)."""
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0).to(_device))
        prob = F.softmax(output, dim=1)[0, 1].item()
    return prob


def predict(image_path: str, lab_values: dict) -> dict:
    """
    Runs the full pipeline: tabular model (via subprocess) + both CNNs + fusion logic.

    Args:
        image_path: path to a blood smear image (any common format PIL can read)
        lab_values: dict with keys wbc_count, hemoglobin, platelet_count,
                    rbc_count, parasitemia_pct

    Returns:
        dict from fuse_prediction(), plus raw model outputs for transparency:
            - "tabular_probs": full 4-class probability array
            - "leukemia_cnn_prob": P(cancer) from the leukemia CNN
            - "malaria_cnn_prob": P(parasitized) from the malaria CNN
    """
    missing = [col for col in FEATURE_COLS if col not in lab_values]
    if missing:
        raise ValueError(f"Missing required lab values: {missing}")

    image = Image.open(image_path).convert("RGB")
    image_tensor = _eval_transform(image)

    tabular_probs_dict = _get_tabular_probs(lab_values)
    tabular_probs = [tabular_probs_dict[name] for name in LABEL_NAMES]

    leukemia_prob = _get_cnn_prob(_leukemia_model, image_tensor)
    malaria_prob = _get_cnn_prob(_malaria_model, image_tensor)

    result = fuse_prediction(
        tabular_probs=tabular_probs,
        leukemia_cnn_prob=leukemia_prob,
        malaria_cnn_prob=malaria_prob,
    )

    result["tabular_probs"] = tabular_probs_dict
    result["leukemia_cnn_prob"] = leukemia_prob
    result["malaria_cnn_prob"] = malaria_prob

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to a blood smear image")
    args = parser.parse_args()

    example_lab_values = {
        "wbc_count": 34.2,
        "hemoglobin": 8.9,
        "platelet_count": 60.0,
        "rbc_count": 3.2,
        "parasitemia_pct": 0.0,
    }

    result = predict(image_path=args.image, lab_values=example_lab_values)
    print("\n--- Prediction result ---")
    for k, v in result.items():
        print(f"{k}: {v}")