"""
Evaluation script for the trained malaria CNN.
Loads the saved model, runs it on the held-out test set, and reports
confusion matrix, sensitivity, specificity, and AUC.
"""

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from torch.utils.data import DataLoader, Subset

from src.data.dataset_malaria import MalariaDataset
from src.data.preprocess_images import get_eval_transforms
from src.models.cnn_malaria import build_malaria_model
from src.training.train_malaria_cnn import get_stratified_splits
from src.evaluation.metrics import get_predictions, print_full_report
from src.utils.config import MALARIA_RAW, SAVED_MODELS_DIR, BATCH_SIZE
from src.utils.device import get_device


def main():
    device = get_device()
    print(f"Using device: {device}")

    raw_dataset = MalariaDataset(root_dir=MALARIA_RAW, transform=None)
    _, _, test_idx = get_stratified_splits(raw_dataset)

    eval_dataset = MalariaDataset(root_dir=MALARIA_RAW, transform=get_eval_transforms())
    test_subset = Subset(eval_dataset, test_idx)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)

    model = build_malaria_model(freeze_backbone=True).to(device)
    model_path = SAVED_MODELS_DIR / "malaria_cnn.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))

    y_true, y_pred, y_probs = get_predictions(model, test_loader, device)

    print_full_report(y_true, y_pred, y_probs, class_names=("Uninfected", "Parasitized"))


if __name__ == "__main__":
    main()