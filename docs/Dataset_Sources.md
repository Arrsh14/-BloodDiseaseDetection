# Dataset Sources

## Image datasets (real data)

**Leukemia — C-NMC 2019**
- Source: Kaggle (C-NMC_Leukemia dataset)
- Classes: `all` (cancer, ALL — Acute Lymphoblastic Leukemia), `hem` (healthy)
- Counts: 7,272 cancer + 3,389 healthy = 10,661 images
- Format: 450×450 `.bmp`

**Malaria — NIH Malaria Cell Images**
- Source: Kaggle (cell-images-for-detecting-malaria dataset), originally
  published by the National Institutes of Health
- Classes: `Parasitized`, `Uninfected`
- Counts: 13,779 + 13,779 = 27,558 images
- Format: variable-sized `.png`

## Tabular dataset (synthetic)

**IMPORTANT: the tabular CBC dataset (`data/raw/tabular/lab_reports.csv`) is
synthetic, not real patient data.** No public dataset exists that combines
leukemia and malaria lab values for the same patient population — such data
is typically private hospital records. Given this gap, a synthetic dataset
was generated programmatically, with value ranges grounded in published
clinical reference ranges rather than arbitrary numbers.

Clinical sources used to define value ranges:

- **Healthy adult CBC reference ranges**: standard clinical reference values
  — WBC 4,500–11,000 cells/µL, hemoglobin 12–16 g/dL, platelets
  150,000–450,000 cells/µL.

- **Leukemia CBC patterns**: based on a published pediatric ALL (Acute
  Lymphoblastic Leukemia) CBC study (PMC6371227) — severe anemia in ~83% of
  cases (median hemoglobin ~7.5 g/dL), severe thrombocytopenia in ~83% of
  cases (median platelets ~47,400/µL), and erratic WBC counts (bimodal:
  either leukocytosis or leukopenia, rarely normal).

- **Malaria hematology patterns**: based on published malaria hematology
  studies (PMC8055386, PMC3910413) — mild anemia (mean hemoglobin
  ~12.0–12.7 g/dL), moderate thrombocytopenia in ~80% of cases (less severe
  than leukemia), and mildly reduced WBC counts.

The generator (`src/data/generate_tabular_data.py`) produced 5,650 records
across 4 classes (normal, leukemia, malaria, both — "both" compounds the
leukemia and malaria patterns). Units used throughout the dataset are
**×10³ cells/µL** for WBC and platelet count (not raw cells/µL) and
**million cells/µL** for RBC count, consistent with standard CBC reporting
conventions.

**Known lesson learned:** early manual testing of the trained model used raw
cells/µL values (e.g., 85000) instead of the dataset's actual ×10³/µL scale,
causing the model to see wildly out-of-distribution inputs and predict
incorrectly. This was not a model bug — verifying input units against
training data ranges is essential before trusting a model's output on
manually-entered test cases.

## Citations

- PMC6371227 — pediatric ALL CBC study (leukemia hematology patterns)
- PMC8055386, PMC3910413 — malaria hematology studies