"""
Direct test of the fusion model's override behavior.

Rather than hunting for naturally-occurring ambiguous lab values (the trained
tabular model turned out to have very sharp, near-deterministic decision
boundaries across the input space we probed), this script calls
fuse_prediction() directly with hand-crafted conflicting inputs: a tabular
signal uncertain between two classes, paired with a strong, confident CNN
signal for one of them. This isolates and demonstrates the fusion model's
learned override behavior directly, without depending on hitting a rare
naturally-ambiguous input by chance.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.models.fusion_model import fuse_prediction

print("=" * 70)
print("TEST 1: Tabular model roughly split between normal/malaria,")
print("        strong malaria CNN signal, weak leukemia CNN signal")
print("=" * 70)
result = fuse_prediction(
    tabular_probs=[0.45, 0.05, 0.40, 0.10],  # ambiguous: normal vs malaria
    leukemia_cnn_prob=0.10,
    malaria_cnn_prob=0.95,
)
for k, v in result.items():
    print(f"{k}: {v}")

print("\n" + "=" * 70)
print("TEST 2: Tabular model roughly split between normal/leukemia,")
print("        strong leukemia CNN signal, weak malaria CNN signal")
print("=" * 70)
result = fuse_prediction(
    tabular_probs=[0.42, 0.38, 0.05, 0.15],  # ambiguous: normal vs leukemia
    leukemia_cnn_prob=0.93,
    malaria_cnn_prob=0.08,
)
for k, v in result.items():
    print(f"{k}: {v}")

print("\n" + "=" * 70)
print("TEST 3 (control): Tabular model VERY confident, CNN disagrees strongly")
print("        — checking fusion doesn't get fooled by a confident tabular call")
print("=" * 70)
result = fuse_prediction(
    tabular_probs=[0.01, 0.97, 0.01, 0.01],  # very confident: leukemia
    leukemia_cnn_prob=0.05,  # CNN strongly disagrees
    malaria_cnn_prob=0.90,
)
for k, v in result.items():
    print(f"{k}: {v}")