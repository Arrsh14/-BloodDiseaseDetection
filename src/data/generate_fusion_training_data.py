"""
Generates training data for the learned fusion model — WITH DELIBERATE AMBIGUITY.

Unlike a naive version, this generator deliberately makes ~40% of samples
"borderline" by blending lab values between the true class and a randomly
chosen other class. This prevents the tabular model from trivially solving
every case (which would teach the fusion model to just copy the tabular
model's argmax, defeating the purpose of fusion). Images remain correctly
labeled ground truth throughout — only lab values are made ambiguous, which
mirrors reality: a lab panel can be borderline, but a directly-observed
cell image is comparatively more definitive.
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Subset

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data.dataset_leukemia import LeukemiaDataset
from src.data.dataset_malaria import MalariaDataset
from src.data.preprocess_images import get_eval_transforms
from src.models.cnn_leukemia import build_leukemia_model_for_inference
from src.models.cnn_malaria import build_malaria_model_for_inference
from src.utils.config import LEUKEMIA_RAW, MALARIA_RAW, SAVED_MODELS_DIR, PROCESSED_DIR, RANDOM_SEED
from src.utils.device import get_device

AMBIGUOUS_FRACTION = 0.4  # 40% of samples get blended/borderline lab values
BLEND_WEIGHT_RANGE = (0.3, 0.5)  # how much of the "other" class bleeds in


def generate_pure_lab_values(diagnosis, rng):
    """Generates lab values purely from one class's clinical distribution."""
    if diagnosis == "normal":
        return {
            "wbc_count": float(np.clip(rng.normal(7.5, 1.5), 4.5, 11.0)),
            "hemoglobin": float(np.clip(rng.normal(14, 1.2), 12, 16)),
            "platelet_count": float(np.clip(rng.normal(300, 70), 150, 450)),
            "rbc_count": float(np.clip(rng.normal(5.0, 0.35), 4.32, 5.72)),
            "parasitemia_pct": 0.0,
        }
    elif diagnosis == "leukemia":
        wbc = rng.normal(45, 30) if rng.random() < 0.5 else rng.normal(2.5, 1.2)
        return {
            "wbc_count": float(np.clip(wbc, 2, 129)),
            "hemoglobin": float(np.clip(rng.normal(7.5, 2.5), 3.5, 15.3)),
            "platelet_count": float(np.clip(rng.gamma(2.0, 25), 5, 150)),
            "rbc_count": float(np.clip(rng.normal(2.8, 0.7), 1.5, 4.3)),
            "parasitemia_pct": 0.0,
        }
    elif diagnosis == "malaria":
        return {
            "wbc_count": float(np.clip(rng.normal(4.8, 1.3), 2, 8)),
            "hemoglobin": float(np.clip(rng.normal(12.3, 1.6), 6.1, 15.2)),
            "platelet_count": float(np.clip(rng.normal(95, 35), 20, 150)),
            "rbc_count": float(np.clip(rng.normal(4.0, 0.5), 2.5, 5.0)),
            "parasitemia_pct": float(np.clip(rng.exponential(3.5), 0.5, 21.5)),
        }
    else:  # both
        return {
            "wbc_count": float(np.clip(rng.normal(35, 25), 2, 129)),
            "hemoglobin": float(np.clip(rng.normal(6.0, 2.0), 3.5, 10.0)),
            "platelet_count": float(np.clip(rng.gamma(1.8, 15), 5, 80)),
            "rbc_count": float(np.clip(rng.normal(2.2, 0.6), 1.5, 3.5)),
            "parasitemia_pct": float(np.clip(rng.exponential(5.0), 1.0, 21.5)),
        }


def generate_lab_values(diagnosis, rng, classes):
    """
    With probability AMBIGUOUS_FRACTION, blends the true class's lab values
    with a randomly chosen OTHER class's lab values — creating a borderline
    case where the lab panel alone doesn't clearly point to one diagnosis.
    The label stays the true diagnosis; only the lab values become noisy.
    """
    pure_values = generate_pure_lab_values(diagnosis, rng)

    if rng.random() < AMBIGUOUS_FRACTION:
        other_classes = [c for c in classes if c != diagnosis]
        other_class = other_classes[rng.integers(len(other_classes))]
        other_values = generate_pure_lab_values(other_class, rng)

        blend_weight = rng.uniform(*BLEND_WEIGHT_RANGE)  # how much "other" bleeds in
        blended = {}
        for key in pure_values:
            blended[key] = (1 - blend_weight) * pure_values[key] + blend_weight * other_values[key]
        return blended

    return pure_values


def main(n_per_class=150):
    device = get_device()
    rng = np.random.default_rng(RANDOM_SEED)
    classes = ["normal", "leukemia", "malaria", "both"]

    leuk_model = build_leukemia_model_for_inference().to(device)
    leuk_model.load_state_dict(torch.load(SAVED_MODELS_DIR / "leukemia_cnn_final.pth", map_location=device))
    leuk_model.eval()

    mal_model = build_malaria_model_for_inference().to(device)
    mal_model.load_state_dict(torch.load(SAVED_MODELS_DIR / "malaria_cnn_final.pth", map_location=device))
    mal_model.eval()

    eval_tf = get_eval_transforms()

    leuk_raw = LeukemiaDataset(root_dir=LEUKEMIA_RAW, transform=None)
    leuk_test_idx = np.load(SAVED_MODELS_DIR / "leukemia_test_indices.npy")
    leuk_eval = LeukemiaDataset(root_dir=LEUKEMIA_RAW, transform=eval_tf)
    leuk_test = Subset(leuk_eval, leuk_test_idx)
    leuk_labels = [leuk_raw.samples[i][1] for i in leuk_test_idx]

    mal_raw = MalariaDataset(root_dir=MALARIA_RAW, transform=None)
    mal_test_idx = np.load(SAVED_MODELS_DIR / "malaria_test_indices.npy")
    mal_eval = MalariaDataset(root_dir=MALARIA_RAW, transform=eval_tf)
    mal_test = Subset(mal_eval, mal_test_idx)
    mal_labels = [mal_raw.samples[i][1] for i in mal_test_idx]

    leuk_cancer_indices = [i for i, l in enumerate(leuk_labels) if l == 1]
    leuk_healthy_indices = [i for i, l in enumerate(leuk_labels) if l == 0]
    mal_parasitized_indices = [i for i, l in enumerate(mal_labels) if l == 1]
    mal_uninfected_indices = [i for i, l in enumerate(mal_labels) if l == 0]

    def cnn_probs(image_tensor):
        input_tensor = image_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            leuk_out = F.softmax(leuk_model(input_tensor), dim=1)[0, 1].item()
            mal_out = F.softmax(mal_model(input_tensor), dim=1)[0, 1].item()
        return leuk_out, mal_out

    records = []
    ambiguous_count = 0

    for diagnosis in classes:
        for _ in range(n_per_class):
            lab_values = generate_lab_values(diagnosis, rng, classes)

            if diagnosis == "leukemia" or (diagnosis == "both" and rng.random() < 0.5):
                idx = leuk_cancer_indices[rng.integers(len(leuk_cancer_indices))]
                image_tensor, _ = leuk_test[idx]
            elif diagnosis == "malaria" or diagnosis == "both":
                idx = mal_parasitized_indices[rng.integers(len(mal_parasitized_indices))]
                image_tensor, _ = mal_test[idx]
            else:  # normal
                if rng.random() < 0.5:
                    idx = leuk_healthy_indices[rng.integers(len(leuk_healthy_indices))]
                    image_tensor, _ = leuk_test[idx]
                else:
                    idx = mal_uninfected_indices[rng.integers(len(mal_uninfected_indices))]
                    image_tensor, _ = mal_test[idx]

            leuk_prob, mal_prob = cnn_probs(image_tensor)

            records.append({
                "wbc_count": lab_values["wbc_count"],
                "hemoglobin": lab_values["hemoglobin"],
                "platelet_count": lab_values["platelet_count"],
                "rbc_count": lab_values["rbc_count"],
                "parasitemia_pct": lab_values["parasitemia_pct"],
                "leukemia_cnn_prob": leuk_prob,
                "malaria_cnn_prob": mal_prob,
                "true_label": diagnosis,
            })

        print(f"Generated {n_per_class} paired samples for class: {diagnosis}")

    df = pd.DataFrame(records)
    out_dir = PROCESSED_DIR / "fusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "fusion_training_data.csv", index=False)
    print(f"\nSaved {len(df)} total paired samples to {out_dir / 'fusion_training_data.csv'}")
    print(f"~{AMBIGUOUS_FRACTION:.0%} of samples have blended/borderline lab values")
    print(df["true_label"].value_counts())


if __name__ == "__main__":
    main()