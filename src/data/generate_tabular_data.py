"""
Generates a synthetic CBC (Complete Blood Count) + lab report dataset for
leukemia / malaria / both / normal classification.

IMPORTANT: This data is SYNTHETIC. Values are sampled from distributions
grounded in real published clinical reference ranges (see docs/Dataset_Sources.md
for citations), but no individual record corresponds to a real patient.
This is intended for demo/portfolio purposes only, not clinical use.

Features generated:
    - wbc_count         : White blood cell count (x10^9/L)
    - hemoglobin        : Hemoglobin (g/dL)
    - platelet_count    : Platelet count (x10^9/L)
    - rbc_count         : Red blood cell count (x10^12/L)
    - parasitemia_pct   : % of RBCs infected with malaria parasite (0 if not malaria)

Label:
    - diagnosis: one of "normal", "leukemia", "malaria", "both"

Reference ranges used (approximate, healthy adult):
    - WBC:        4.0 - 11.0 x10^9/L
    - Hemoglobin: 13.5 - 17.5 g/dL (male range used as baseline; simplified, not sex-split)
    - Platelets:  150 - 450 x10^9/L
    - RBC:        4.5 - 5.9 x10^12/L

Leukemia (ALL/AML) typically shows:
    - WBC often markedly elevated (can range widely, sometimes low in some subtypes)
    - Hemoglobin reduced (anemia common)
    - Platelets reduced (thrombocytopenia common)

Malaria typically shows:
    - WBC normal or mildly reduced
    - Hemoglobin reduced (hemolytic anemia)
    - Platelets reduced (thrombocytopenia very common in malaria)
    - Parasitemia % present and >0
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import TABULAR_RAW, RANDOM_SEED

N_PER_CLASS = 1500  # records per class -> 6000 total across 4 classes


def generate_normal(n, rng):
    return pd.DataFrame({
        "wbc_count": rng.normal(7.5, 1.5, n).clip(4.0, 11.0),
        "hemoglobin": rng.normal(15.0, 1.2, n).clip(12.0, 17.5),
        "platelet_count": rng.normal(300, 60, n).clip(150, 450),
        "rbc_count": rng.normal(5.1, 0.4, n).clip(4.2, 6.1),
        "parasitemia_pct": np.zeros(n),
        "diagnosis": "normal",
    })


def generate_leukemia(n, rng):
    # WBC is bimodal in real leukemia data (some subtypes present low, most present high) —
    # simplified here as skewed high with a long tail, which is the more common presentation.
    wbc = rng.gamma(shape=2.0, scale=15.0, size=n) + 4.0  # skewed high, min ~4
    return pd.DataFrame({
        "wbc_count": wbc.clip(1.0, 200.0),
        "hemoglobin": rng.normal(9.0, 1.8, n).clip(4.0, 13.0),       # anemia
        "platelet_count": rng.normal(60, 40, n).clip(5, 150),        # thrombocytopenia
        "rbc_count": rng.normal(3.2, 0.6, n).clip(1.5, 4.3),
        "parasitemia_pct": np.zeros(n),
        "diagnosis": "leukemia",
    })


def generate_malaria(n, rng):
    return pd.DataFrame({
        "wbc_count": rng.normal(6.0, 2.0, n).clip(2.0, 12.0),        # normal to mildly low
        "hemoglobin": rng.normal(10.5, 1.8, n).clip(5.0, 14.0),      # hemolytic anemia
        "platelet_count": rng.normal(90, 45, n).clip(10, 150),       # thrombocytopenia common
        "rbc_count": rng.normal(4.0, 0.5, n).clip(2.5, 5.0),
        "parasitemia_pct": rng.gamma(shape=1.5, scale=2.0, size=n).clip(0.01, 30.0),
        "diagnosis": "malaria",
    })


def generate_both(n, rng):
    # Blend: leukemia-pattern WBC/platelets + malaria-pattern parasitemia/hemoglobin.
    # Represents a patient presenting with both conditions concurrently.
    wbc = rng.gamma(shape=2.0, scale=13.0, size=n) + 4.0
    return pd.DataFrame({
        "wbc_count": wbc.clip(1.0, 180.0),
        "hemoglobin": rng.normal(8.0, 1.8, n).clip(3.5, 12.0),       # worse anemia, both contribute
        "platelet_count": rng.normal(50, 35, n).clip(5, 130),        # worse thrombocytopenia
        "rbc_count": rng.normal(3.0, 0.6, n).clip(1.5, 4.2),
        "parasitemia_pct": rng.gamma(shape=1.5, scale=2.0, size=n).clip(0.01, 30.0),
        "diagnosis": "both",
    })


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    df_normal = generate_normal(N_PER_CLASS, rng)
    df_leukemia = generate_leukemia(N_PER_CLASS, rng)
    df_malaria = generate_malaria(N_PER_CLASS, rng)
    df_both = generate_both(N_PER_CLASS, rng)

    df = pd.concat([df_normal, df_leukemia, df_malaria, df_both], ignore_index=True)

    # Round for readability, shuffle rows so classes aren't in contiguous blocks
    for col in ["wbc_count", "hemoglobin", "platelet_count", "rbc_count", "parasitemia_pct"]:
        df[col] = df[col].round(2)
    df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    TABULAR_RAW.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABULAR_RAW, index=False)

    print(f"Saved {len(df)} records to {TABULAR_RAW}")
    print(df["diagnosis"].value_counts())
    print(df.head())


if __name__ == "__main__":
    main()