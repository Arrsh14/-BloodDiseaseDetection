"""
ResNet18 transfer learning model for leukemia classification (cancer vs healthy).
"""

import torch.nn as nn
from torchvision import models


def build_leukemia_model(unfreeze_layer4=False):
    """
    Loads a pretrained ResNet18 and adapts it for binary classification.

    Args:
        unfreeze_layer4: if False (phase 1), freezes the entire backbone so
                          only the new fc layer trains. If True (phase 2),
                          also unfreezes layer4 (the last conv block) for
                          fine-tuning, while layer1-3 stay frozen.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Phase 2: unfreeze just layer4 for fine-tuning
    if unfreeze_layer4:
        for param in model.layer4.parameters():
            param.requires_grad = True

    # Replace final fully-connected layer: 512 -> 2 classes (cancer/healthy)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    # New fc layer has requires_grad=True by default regardless of the freezing above,
    # since it's a fresh layer created after the freeze loop.

    return model


if __name__ == "__main__":
    model = build_leukemia_model()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Phase 1 - Trainable params: {trainable:,} / {total:,}")

    model_phase2 = build_leukemia_model(unfreeze_layer4=True)
    trainable2 = sum(p.numel() for p in model_phase2.parameters() if p.requires_grad)
    print(f"Phase 2 - Trainable params: {trainable2:,} / {total:,}")