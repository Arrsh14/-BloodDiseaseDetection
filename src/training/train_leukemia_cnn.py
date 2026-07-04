"""
Trains the leukemia CNN (cancer vs healthy) using two-phase transfer learning:
    Phase 1: backbone frozen, only the new fc layer trains
    Phase 2: fc + layer4 unfrozen, fine-tuned at a lower learning rate

Handles:
    - Stratified train/val/test split (since raw data is only split into all/hem)
    - Class imbalance (cancer ~68%, healthy ~32%) via weighted loss
    - Checkpointing best model (by validation accuracy) for each phase
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.data.dataset_leukemia import LeukemiaDataset
from src.data.preprocess_images import get_train_transforms, get_eval_transforms
from src.models.cnn_leukemia import build_leukemia_model
from src.utils.config import LEUKEMIA_RAW, SAVED_MODELS_DIR, BATCH_SIZE, RANDOM_SEED
from src.utils.device import get_device

PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 10
PHASE1_LR = 1e-3
PHASE2_LR = 1e-5  # much lower — we're fine-tuning pretrained conv weights, not training from scratch


def stratified_split_indices(labels, test_size, val_size, seed):
    """
    Splits indices into train/val/test, preserving class balance in each split.
    labels: list/array of integer labels corresponding to dataset order.
    """
    indices = np.arange(len(labels))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, stratify=labels, random_state=seed
    )
    train_val_labels = np.array(labels)[train_val_idx]
    relative_val_size = val_size / (1 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=relative_val_size, stratify=train_val_labels, random_state=seed
    )
    return train_idx, val_idx, test_idx


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    """Runs one epoch of training or evaluation. Returns (avg_loss, accuracy)."""
    model.train() if train else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    device = get_device()
    print(f"Using device: {device}")

    # --- Build base (unlabeled-transform) dataset just to get labels for stratified split ---
    base_dataset = LeukemiaDataset(root_dir=LEUKEMIA_RAW, transform=None)
    labels = [label for _, label in base_dataset.samples]

    train_idx, val_idx, test_idx = stratified_split_indices(
        labels, test_size=0.15, val_size=0.15, seed=RANDOM_SEED
    )
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    # --- Separate dataset instances per split, each with the right transform ---
    train_dataset_full = LeukemiaDataset(root_dir=LEUKEMIA_RAW, transform=get_train_transforms())
    eval_dataset_full = LeukemiaDataset(root_dir=LEUKEMIA_RAW, transform=get_eval_transforms())

    train_ds = Subset(train_dataset_full, train_idx)
    val_ds = Subset(eval_dataset_full, val_idx)
    test_ds = Subset(eval_dataset_full, test_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- Class weights to handle imbalance (cancer=1 majority, healthy=0 minority) ---
    train_labels = np.array(labels)[train_idx]
    class_counts = np.bincount(train_labels)  # [count_healthy, count_cancer]
    class_weights = torch.tensor(
        [len(train_labels) / (2.0 * c) for c in class_counts], dtype=torch.float32
    ).to(device)
    print(f"Class counts (train): healthy={class_counts[0]}, cancer={class_counts[1]}")
    print(f"Class weights: {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ============ PHASE 1: train only fc, backbone frozen ============
    print("\n=== Phase 1: training fc layer only (backbone frozen) ===")
    model = build_leukemia_model(unfreeze_layer4=False).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=PHASE1_LR)

    best_val_acc = 0.0
    for epoch in range(PHASE1_EPOCHS):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f"[Phase1 Epoch {epoch+1}/{PHASE1_EPOCHS}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVED_MODELS_DIR / "leukemia_cnn_phase1_best.pth")

    # ============ PHASE 2: unfreeze layer4, fine-tune ============
    print("\n=== Phase 2: fine-tuning fc + layer4 ===")
    model = build_leukemia_model(unfreeze_layer4=True).to(device)
    # Load phase 1's best weights so phase 2 builds on what was already learned
    model.load_state_dict(torch.load(SAVED_MODELS_DIR / "leukemia_cnn_phase1_best.pth", map_location=device))

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=PHASE2_LR)

    best_val_acc = 0.0
    for epoch in range(PHASE2_EPOCHS):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f"[Phase2 Epoch {epoch+1}/{PHASE2_EPOCHS}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVED_MODELS_DIR / "leukemia_cnn_final.pth")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Final model saved to: {SAVED_MODELS_DIR / 'leukemia_cnn_final.pth'}")

    # --- Final check on test set using the best phase-2 model ---
    model.load_state_dict(torch.load(SAVED_MODELS_DIR / "leukemia_cnn_final.pth", map_location=device))
    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    print(f"\nTest accuracy (quick check, full metrics in evaluate_leukemia.py): {test_acc:.4f}")

    # Save test indices so evaluate_leukemia.py uses the EXACT same test set, not a new random split
    np.save(SAVED_MODELS_DIR / "leukemia_test_indices.npy", test_idx)


if __name__ == "__main__":
    main()