"""
ResNet18 transfer learning model for leukemia classification (cancer vs healthy).
"""

import torch.nn as nn
from torchvision import models


def build_leukemia_model(freeze_backbone=True):
    """
    Loads a pretrained ResNet18 and adapts it for binary classification.

    Args:
        freeze_backbone: if True, freezes all conv layers so only the new
                          final layer trains initially. Set False later to
                          fine-tune deeper layers at a lower learning rate.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace final fully-connected layer: 512 -> 2 classes (cancer/healthy)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    # Only the new fc layer has requires_grad=True by default (fresh layer),
    # so it will train even with the backbone frozen.

    return model


if __name__ == "__main__":
    model = build_leukemia_model()
    print(model)