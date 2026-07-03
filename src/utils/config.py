from pathlib import Path

# Root paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAVED_MODELS_DIR = ROOT_DIR / "saved_models"
RESULTS_DIR = ROOT_DIR / "results"

# Leukemia
LEUKEMIA_RAW = RAW_DIR / "leukemia"
LEUKEMIA_PROCESSED = PROCESSED_DIR / "leukemia"

# Malaria
MALARIA_RAW = RAW_DIR / "malaria"
MALARIA_PROCESSED = PROCESSED_DIR / "malaria"

# Tabular
TABULAR_RAW = RAW_DIR / "tabular" / "lab_reports.csv"
TABULAR_PROCESSED = PROCESSED_DIR / "tabular" / "cleaned_lab_data.csv"

# Hyperparameters (placeholder, tune later)
IMAGE_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 20
RANDOM_SEED = 42