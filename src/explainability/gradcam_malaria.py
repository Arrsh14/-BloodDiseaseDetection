"""
Grad-CAM visualization for the malaria CNN.
Shows which regions of a cell image the model focused on when predicting
parasitized vs uninfected, by hooking into layer4.
"""

import torch
import numpy as np
import cv2

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from torch.utils.data import Subset

from src.data.dataset_malaria import MalariaDataset
from src.data.preprocess_images import get_eval_transforms
from src.models.cnn_malaria import build_malaria_model
from src.training.train_malaria_cnn import get_stratified_splits
from src.explainability.gradcam_leukemia import GradCAM, denormalize_image, overlay_heatmap
from src.utils.config import MALARIA_RAW, SAVED_MODELS_DIR, ROOT_DIR
from src.utils.device import get_device


def main(num_samples=8):
    device = get_device()
    print(f"Using device: {device}")

    raw_dataset = MalariaDataset(root_dir=MALARIA_RAW, transform=None)
    _, _, test_idx = get_stratified_splits(raw_dataset)

    eval_dataset = MalariaDataset(root_dir=MALARIA_RAW, transform=get_eval_transforms())
    test_subset = Subset(eval_dataset, test_idx)

    model = build_malaria_model(freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(SAVED_MODELS_DIR / "malaria_cnn.pth", map_location=device))
    model.eval()

    gradcam = GradCAM(model, target_layer=model.layer4)

    output_dir = ROOT_DIR / "results" / "gradcam_outputs" / "malaria"
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = {0: "Uninfected", 1: "Parasitized"}

    for i in range(num_samples):
        image_tensor, true_label = test_subset[i]
        input_tensor = image_tensor.unsqueeze(0).to(device)

        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()

        cam = gradcam.generate(input_tensor, class_idx=pred_class)
        original_img = denormalize_image(image_tensor)
        overlayed = overlay_heatmap(original_img, cam)

        save_path = output_dir / f"sample_{i}_true-{class_names[true_label]}_pred-{class_names[pred_class]}.png"
        cv2.imwrite(str(save_path), cv2.cvtColor(np.uint8(overlayed * 255), cv2.COLOR_RGB2BGR))
        print(f"Saved: {save_path.name}")

    print(f"\nDone. {num_samples} Grad-CAM images saved to {output_dir}")


if __name__ == "__main__":
    main()