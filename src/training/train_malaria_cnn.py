"""
Training script for the malaria CNN (ResNet18 transfer learning).
Dataset is balanced (13,779 / 13,779), so no class weighting needed here —
unlike leukemia, plain CrossEntropyLoss is fine.
Phase 1: train only the final fc layer (backbone frozen).
Phase 2: unfreeze layer4 (last conv block) and fine-tune at a lower LR.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data.dataset_malaria import MalariaDataset
from src.data.preprocess_images import get_train_transforms, get_eval_transforms
from src.models.cnn_malaria import build_malaria_model
from src.utils.config import MALARIA_RAW, SAVED_MODELS_DIR, BATCH_SIZE, LEARNING_RATE, RANDOM_SEED
from src.utils.device import get_device


def get_stratified_splits(dataset, val_size=0.15, test_size=0.15, seed=RANDOM_SEED):
    """
    Splits dataset indices into train/val/test, preserving class ratio in each split.
    Dataset is balanced here, but stratifying is still good practice — costs nothing,
    guarantees exact 50/50 in every split rather than leaving it to chance.
    """
    labels = [label for _, label in dataset.samples]
    indices = list(range(len(dataset)))

    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, stratify=labels, random_state=seed
    )

    train_val_labels = [labels[i] for i in train_val_idx]
    relative_val_size = val_size / (1 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=relative_val_size, stratify=train_val_labels, random_state=seed
    )

    return train_idx, val_idx, test_idx


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def run_training_phase(model, train_loader, val_loader, criterion, optimizer,
                        device, num_epochs, save_path, best_val_acc, phase_name):
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"[{phase_name}] Epoch {epoch}/{num_epochs} | "
              f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
              f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best model saved (val_acc: {val_acc:.4f})")

    return best_val_acc


def main():
    device = get_device()
    print(f"Using device: {device}")

    raw_dataset = MalariaDataset(root_dir=MALARIA_RAW, transform=None)
    train_idx, val_idx, test_idx = get_stratified_splits(raw_dataset)
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    train_dataset = MalariaDataset(root_dir=MALARIA_RAW, transform=get_train_transforms())
    eval_dataset = MalariaDataset(root_dir=MALARIA_RAW, transform=get_eval_transforms())

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(eval_dataset, val_idx)
    test_subset = Subset(eval_dataset, test_idx)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)

    model = build_malaria_model(freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()  # no class weights — dataset is balanced

    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = SAVED_MODELS_DIR / "malaria_cnn.pth"
    best_val_acc = 0.0

    # ---- PHASE 1: train only the fc layer (backbone frozen) ----
    optimizer_phase1 = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    best_val_acc = run_training_phase(
        model, train_loader, val_loader, criterion, optimizer_phase1,
        device, num_epochs=10, save_path=save_path,
        best_val_acc=best_val_acc, phase_name="Phase 1 (frozen)"
    )

    # ---- PHASE 2: unfreeze layer4, fine-tune at lower LR ----
    print("\nUnfreezing layer4 for fine-tuning...\n")
    for param in model.layer4.parameters():
        param.requires_grad = True

    fine_tune_lr = LEARNING_RATE * 0.1
    optimizer_phase2 = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=fine_tune_lr
    )

    best_val_acc = run_training_phase(
        model, train_loader, val_loader, criterion, optimizer_phase2,
        device, num_epochs=5, save_path=save_path,
        best_val_acc=best_val_acc, phase_name="Phase 2 (fine-tune)"
    )

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {save_path}")

    model.load_state_dict(torch.load(save_path))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test accuracy (best model): {test_acc:.4f}")


if __name__ == "__main__":
    main()