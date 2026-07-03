"""
Preprocesses the synthetic CBC lab report dataset:
    1. Loads the raw CSV (data/raw/tabular/lab_reports.csv)
    2. Encodes the 'diagnosis' label (text -> integer)
    3. Splits into train / val / test (70 / 15 / 15), stratified by diagnosis
       so each split preserves the original class balance
    4. Saves splits + the label encoding mapping to data/processed/tabular/

Note: features are left unscaled here. XGBoost (tree-based) does not require
feature scaling. If a neural model is later added on the tabular data, scale
at that point using stats computed from the TRAIN split only (to avoid leakage).
"""

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import TABULAR_RAW, PROCESSED_DIR, RANDOM_SEED

TABULAR_PROCESSED_DIR = PROCESSED_DIR / "tabular"

# Diagnosis label encoding — fixed order so it's stable and reproducible
LABEL_MAP = {
    "normal": 0,
    "leukemia": 1,
    "malaria": 2,
    "both": 3,
}


def main():
    df = pd.read_csv(TABULAR_RAW)

    # Encode labels
    df["label"] = df["diagnosis"].map(LABEL_MAP)
    if df["label"].isnull().any():
        raise ValueError("Found diagnosis values not in LABEL_MAP — check the raw CSV.")

    # First split off test (15%), then split remaining into train/val (70/15 of original)
    # stratify=df['label'] ensures each split keeps the same class proportions
    train_val_df, test_df = train_test_split(
        df, test_size=0.15, stratify=df["label"], random_state=RANDOM_SEED
    )
    # 0.15 / (1 - 0.15) = 0.1765 -> gives val = 15% of original total
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.1765, stratify=train_val_df["label"], random_state=RANDOM_SEED
    )

    TABULAR_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TABULAR_PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(TABULAR_PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(TABULAR_PROCESSED_DIR / "test.csv", index=False)

    with open(TABULAR_PROCESSED_DIR / "label_map.json", "w") as f:
        json.dump(LABEL_MAP, f, indent=2)

    print(f"Train: {len(train_df)} rows -> {TABULAR_PROCESSED_DIR / 'train.csv'}")
    print(f"Val:   {len(val_df)} rows -> {TABULAR_PROCESSED_DIR / 'val.csv'}")
    print(f"Test:  {len(test_df)} rows -> {TABULAR_PROCESSED_DIR / 'test.csv'}")
    print()
    print("Class balance check (should be ~consistent across splits):")
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        pct = (split_df["diagnosis"].value_counts(normalize=True) * 100).round(1)
        print(f"\n{name}:")
        print(pct)


if __name__ == "__main__":
    main()