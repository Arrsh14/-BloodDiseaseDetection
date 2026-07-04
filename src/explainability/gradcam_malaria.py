"""
Grad-CAM visualization for the malaria CNN.

Same approach as gradcam_leukemia.py: saves heatmaps for true positives,
true negatives, false positives, and false negatives so we can visually
inspect whether the model's attention is on the parasite/cell body,
or elsewhere.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.data.dataset_malaria import MalariaDataset
from src.data.preprocess_images import get_eval_transforms, IMAGENET_MEAN, IMAGENET_STD
from src.models.cnn_malaria import build_malaria_model
from src.utils.config import MALARIA_RAW, SAVED_MODELS_DIR, RESULTS_DIR
from src.utils.device import get_device

N_EXAMPLES_PER_CATEGORY = 3


def denormalize(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor * std + mean
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def main():
    device = get_device()
    print(f"Using device: {device}")

    test_idx = np.load(SAVED_MODELS_DIR / "malaria_test_indices.npy")
    eval_dataset_full = MalariaDataset(root_dir=MALARIA_RAW, transform=get_eval_transforms())
    test_ds = Subset(eval_dataset_full, test_idx)

    model = build_malaria_model(unfreeze_layer4=True).to(device)
    model.load_state_dict(torch.load(SAVED_MODELS_DIR / "malaria_cnn_final.pth", map_location=device))
    model.eval()

    target_layer = model.layer4[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])

    print("Running predictions to categorize examples...")
    records = []
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
        "true_positive_parasitized": find_examples(1, 1, N_EXAMPLES_PER_CATEGORY),
        "true_negative_uninfected": find_examples(0, 0, N_EXAMPLES_PER_CATEGORY),
        "false_positive": find_examples(0, 1, N_EXAMPLES_PER_CATEGORY),
        "false_negative": find_examples(1, 0, N_EXAMPLES_PER_CATEGORY),
    }

    out_dir = RESULTS_DIR / "gradcam_output" / "malaria"
    out_dir.mkdir(parents=True, exist_ok=True)

    for category, examples in categories.items():
        if len(examples) == 0:
            print(f"No examples found for category: {category}")
            continue

        for rank, (idx, true_label, pred_label) in enumerate(examples):
            image, _ = test_ds[idx]
            input_tensor = image.unsqueeze(0).to(device)

            grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_label)])
            grayscale_cam = grayscale_cam[0, :]

            rgb_img = denormalize(image.cpu())
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(rgb_img)
            axes[0].set_title("Original")
            axes[0].axis("off")
            axes[1].imshow(visualization)
            axes[1].set_title("Grad-CAM")
            axes[1].axis("off")

            label_names = {0: "uninfected", 1: "parasitized"}
            fig.suptitle(
                f"{category} | true={label_names[true_label]}, pred={label_names[pred_label]}"
            )
            plt.tight_layout()

            out_path = out_dir / f"{category}_{rank}.png"
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"Saved {out_path}")

    print(f"\nAll Grad-CAM visualizations saved to {out_dir}")
    print("Inspect: does the highlighted region overlap with the visible parasite/cell body, "
          "or track background/edge artifacts?")


if __name__ == "__main__":
    main()