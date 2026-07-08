"""
End-to-end evaluation of the complete prediction pipeline.

Generates a held-out test set (different random seed than fusion training
data, so these are genuinely unseen cases) of paired (image, lab values,
true label) samples — WITH DELIBERATE AMBIGUITY injected into ~40% of lab
values (same technique used in generate_fusion_training_data.py, needed to
avoid the tabular model trivially solving every case) — runs each one
through the ACTUAL predict() function (the same one the Streamlit app uses:
tabular subprocess + both CNNs + learned fusion model, end to end), and
reports overall accuracy and a confusion matrix for the complete system as
a user would experience it.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data.dataset_leukemia import LeukemiaDataset
from src.data.dataset_malaria import MalariaDataset
from src.utils.config import LEUKEMIA_RAW, MALARIA_RAW, SAVED_MODELS_DIR
from src.inference.predict import predict

N_PER_CLASS = 30  # 30 x 4 classes = 120 total end-to-end test cases
EVAL_SEED = 999  # DIFFERENT from RANDOM_SEED used everywhere else — genuinely held-out
AMBIGUOUS_FRACTION = 0.4
BLEND_WEIGHT_RANGE = (0.3, 0.5)


def generate_pure_lab_values(diagnosis, rng):
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
    """Blends ~40% of samples with another class's distribution — real ambiguity."""
    pure_values = generate_pure_lab_values(diagnosis, rng)
    if rng.random() < AMBIGUOUS_FRACTION:
        other_classes = [c for c in classes if c != diagnosis]
        other_class = other_classes[rng.integers(len(other_classes))]
        other_values = generate_pure_lab_values(other_class, rng)
        blend_weight = rng.uniform(*BLEND_WEIGHT_RANGE)
        blended = {}
        for key in pure_values:
            blended[key] = (1 - blend_weight) * pure_values[key] + blend_weight * other_values[key]
        return blended
    return pure_values


def main():
    rng = np.random.default_rng(EVAL_SEED)
    classes = ["normal", "leukemia", "malaria", "both"]

    leuk_raw = LeukemiaDataset(root_dir=LEUKEMIA_RAW, transform=None)
    leuk_test_idx = np.load(SAVED_MODELS_DIR / "leukemia_test_indices.npy")
    leuk_cancer_paths = [leuk_raw.samples[i][0] for i in leuk_test_idx if leuk_raw.samples[i][1] == 1]
    leuk_healthy_paths = [leuk_raw.samples[i][0] for i in leuk_test_idx if leuk_raw.samples[i][1] == 0]

    mal_raw = MalariaDataset(root_dir=MALARIA_RAW, transform=None)
    mal_test_idx = np.load(SAVED_MODELS_DIR / "malaria_test_indices.npy")
    mal_parasitized_paths = [mal_raw.samples[i][0] for i in mal_test_idx if mal_raw.samples[i][1] == 1]
    mal_uninfected_paths = [mal_raw.samples[i][0] for i in mal_test_idx if mal_raw.samples[i][1] == 0]

    results = []

    for diagnosis in classes:
        for _ in range(N_PER_CLASS):
            lab_values = generate_lab_values(diagnosis, rng, classes)

            if diagnosis == "leukemia" or (diagnosis == "both" and rng.random() < 0.5):
                image_path = leuk_cancer_paths[rng.integers(len(leuk_cancer_paths))]
            elif diagnosis == "malaria" or diagnosis == "both":
                image_path = mal_parasitized_paths[rng.integers(len(mal_parasitized_paths))]
            else:  # normal
                if rng.random() < 0.5:
                    image_path = leuk_healthy_paths[rng.integers(len(leuk_healthy_paths))]
                else:
                    image_path = mal_uninfected_paths[rng.integers(len(mal_uninfected_paths))]

            try:
                pred_result = predict(image_path=str(image_path), lab_values=lab_values)
                predicted = pred_result["diagnosis"]
            except Exception as e:
                print(f"Error on {image_path}: {e}")
                predicted = "ERROR"

            results.append({"true_label": diagnosis, "predicted": predicted})

        print(f"Evaluated {N_PER_CLASS} cases for class: {diagnosis}")

    df = pd.DataFrame(results)

    out_dir = SAVED_MODELS_DIR.parent / "results" / "end_to_end_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "end_to_end_results.csv", index=False)

    acc = accuracy_score(df["true_label"], df["predicted"])
    print(f"\n{'='*60}")
    print(f"END-TO-END SYSTEM ACCURACY: {acc:.4f} ({len(df)} held-out cases)")
    print(f"{'='*60}\n")
    print("Classification report:")
    print(classification_report(df["true_label"], df["predicted"]))
    print("\nConfusion matrix (rows=true, cols=pred):")
    labels_order = sorted(df["true_label"].unique())
    print(f"Classes order: {labels_order}")
    print(confusion_matrix(df["true_label"], df["predicted"], labels=labels_order))
    print(f"\nFull results saved to: {out_dir / 'end_to_end_results.csv'}")


if __name__ == "__main__":
    main()