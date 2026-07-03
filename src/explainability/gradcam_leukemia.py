"""
Grad-CAM visualization for the leukemia CNN.
Shows which regions of a cell image the model focused on when predicting
cancer vs healthy, by hooking into layer4 (the last conv block).
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from torch.utils.data import DataLoader, Subset

from src.data.dataset_leukemia import LeukemiaDataset
from src.data.preprocess_images import get_eval_transforms, IMAGENET_MEAN, IMAGENET_STD
from src.models.cnn_leukemia import build_leukemia_model
from src.training.train_leukemia_cnn import get_stratified_splits
from src.utils.config import LEUKEMIA_RAW, SAVED_MODELS_DIR, ROOT_DIR
from src.utils.device import get_device


class GradCAM:
    """
    Hooks into a target conv layer, captures its activations and gradients
    during a forward+backward pass, and combines them into a heatmap showing
    which spatial regions most influenced the model's prediction.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Hooks capture the layer's output (forward) and its gradient (backward)
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        """
        Runs a forward + backward pass for the given class, then computes
        the Grad-CAM heatmap.
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        # Backprop from the score of the class we're explaining
        score = output[0, class_idx]
        score.backward()

        # Global-average-pool the gradients -> one importance weight per channel
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted sum of activation channels -> single-channel importance map
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # only keep positive influence, discard negative

        # Resize to match input image size, normalize to 0-1
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


def denormalize_image(tensor):
    """Reverses ImageNet normalization so we can display the original-looking image."""
    img = tensor.clone().cpu().numpy().transpose(1, 2, 0)
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img


def overlay_heatmap(image, cam, alpha=0.4):
    """Overlays a Grad-CAM heatmap on top of the original image."""
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

    overlay = heatmap * alpha + image * (1 - alpha)
    return np.clip(overlay, 0, 1)

def main(num_samples=8):
    device = get_device()
    print(f"Using device: {device}")

    raw_dataset = LeukemiaDataset(root_dir=LEUKEMIA_RAW, transform=None)
    _, _, test_idx = get_stratified_splits(raw_dataset)

    eval_dataset = LeukemiaDataset(root_dir=LEUKEMIA_RAW, transform=get_eval_transforms())
    test_subset = Subset(eval_dataset, test_idx)

    model = build_leukemia_model(freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(SAVED_MODELS_DIR / "leukemia_cnn.pth", map_location=device))
    model.eval()

    # Two GradCAM instances: one per layer, so we can compare resolution/accuracy
    gradcam_layer3 = GradCAM(model, target_layer=model.layer3)
    gradcam_layer4 = GradCAM(model, target_layer=model.layer4)

    output_dir = ROOT_DIR / "results" / "gradcam_outputs" / "leukemia_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = {0: "Healthy", 1: "Cancer"}

    for i in range(num_samples):
        image_tensor, true_label = test_subset[i]
        input_tensor = image_tensor.unsqueeze(0).to(device)

        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()

        original_img = denormalize_image(image_tensor)

        # layer3 heatmap
        cam3 = gradcam_layer3.generate(input_tensor, class_idx=pred_class)
        overlay3 = overlay_heatmap(original_img, cam3)

        # layer4 heatmap (need fresh forward pass since backward() was already called)
        output = model(input_tensor)
        cam4 = gradcam_layer4.generate(input_tensor, class_idx=pred_class)
        overlay4 = overlay_heatmap(original_img, cam4)

        # Save side by side: original | layer3 | layer4
        combined = np.concatenate([original_img, overlay3, overlay4], axis=1)

        save_path = output_dir / f"sample_{i}_true-{class_names[true_label]}_pred-{class_names[pred_class]}.png"
        cv2.imwrite(str(save_path), cv2.cvtColor(np.uint8(combined * 255), cv2.COLOR_RGB2BGR))
        print(f"Saved: {save_path.name}")

    print(f"\nDone. {num_samples} comparison images saved to {output_dir}")
    print("Each image shows: [original | layer3 heatmap | layer4 heatmap]")


if __name__ == "__main__":
    main()