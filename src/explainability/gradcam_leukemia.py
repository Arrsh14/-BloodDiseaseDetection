"""
Grad-CAM visualization for the leukemia CNN.

Generates heatmaps showing which regions of each cell image the model
focused on to make its prediction. Uses the pytorch-grad-cam library,
targeting the last conv block (layer4) of ResNet18.

Saves a mix of:
    - Correctly classified examples (both classes)
    - Misclassified examples (both false positive and false negative)
so we can visually inspect whether errors correlate with attention on
irrelevant regions (e.g., cell edges/background rather than interior).
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.data.dataset_leukemia import LeukemiaDataset
from src.data.preprocess_images import get_eval_transforms, IMAGENET_MEAN, IMAGENET_STD
from src.models.cnn_leukemia import build_leukemia_model
from src.utils.config import LEUKEMIA_RAW, SAVED_MODELS_DIR, RESULTS_DIR
from src.utils.device import get_device

N_EXAMPLES_PER_CATEGORY = 3  # how many images to save per category below


def denormalize(tensor):
    """Reverses ImageNet normalization so the image can be displayed correctly."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor * std + mean
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()  # CHW -> HWC for plotting/cam overlay


def main():
    device = get_device()
    print(f"Using device: {device}")

    test_idx = np.load(SAVED_MODELS_DIR / "leukemia_test_indices.npy")
    eval_dataset_full = LeukemiaDataset(root_dir=LEUKEMIA_RAW, transform=get_eval_transforms())
    test_ds = Subset(eval_dataset_full, test_idx)

    model = build_leukemia_model(unfreeze_layer4=True).to(device)
    model.load_state_dict(torch.load(SAVED_MODELS_DIR / "leukemia_cnn_final.pth", map_location=device))
    model.eval()

    # Target the last conv block for Grad-CAM — standard choice for ResNet architectures,
    # since layer4 captures the highest-level spatial features before global pooling.
    target_layer = model.layer4[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])

    # --- First pass: get predictions for every test example, to sort into categories ---
    print("Running predictions to categorize examples...")
    records = []  # list of (index_in_test_ds, true_label, pred_label)
    with torch.no_grad():
        for i in range(len(test_ds)):
            image, label = test_ds[i]
            output = model(image.unsqueeze(0).to(device))
            pred = output.argmax(dim=1).item()
            records.append((i, label, pred))

    def find_examples(true_label, pred_label, n):
        matches = [r for r in records if r[1] == true_label and r[2] == pred_label]
        return matches[:n]

    categories = {
        "true_positive_cancer": find_examples(1, 1, N_EXAMPLES_PER_CATEGORY),   # correctly caught cancer
        "true_negative_healthy": find_examples(0, 0, N_EXAMPLES_PER_CATEGORY),  # correctly cleared healthy
        "false_positive": find_examples(0, 1, N_EXAMPLES_PER_CATEGORY),        # healthy misclassified as cancer
        "false_negative": find_examples(1, 0, N_EXAMPLES_PER_CATEGORY),        # cancer misclassified as healthy
    }

    out_dir = RESULTS_DIR / "gradcam_output" / "leukemia"
    out_dir.mkdir(parents=True, exist_ok=True)

    for category, examples in categories.items():
        if len(examples) == 0:
            print(f"No examples found for category: {category}")
            continue

        for rank, (idx, true_label, pred_label) in enumerate(examples):
            image, _ = test_ds[idx]
            input_tensor = image.unsqueeze(0).to(device)

            grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_label)])
            grayscale_cam = grayscale_cam[0, :]  # first (only) image in batch

            rgb_img = denormalize(image.cpu())
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(rgb_img)
            axes[0].set_title("Original")
            axes[0].axis("off")
            axes[1].imshow(visualization)
            axes[1].set_title("Grad-CAM")
            axes[1].axis("off")

            label_names = {0: "healthy", 1: "cancer"}
            fig.suptitle(
                f"{category} | true={label_names[true_label]}, pred={label_names[pred_label]}"
            )
            plt.tight_layout()

            out_path = out_dir / f"{category}_{rank}.png"
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"Saved {out_path}")

    print(f"\nAll Grad-CAM visualizations saved to {out_dir}")
    print("Inspect these images: does the highlighted (red/warm) region overlap with the "
          "actual cell body/interior, or does it mostly track edges/background artifacts?")


if __name__ == "__main__":
    main()