"""
Shared image preprocessing transforms for the leukemia and malaria CNNs.
Both use ResNet18 pretrained on ImageNet, so images must be resized to
224x224 and normalized using ImageNet's mean/std statistics.
"""

from torchvision import transforms

# Standard ImageNet normalization stats — required since we're using
# ImageNet-pretrained ResNet18 weights.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_SIZE = 224


def get_train_transforms():
    """
    Transform pipeline for TRAINING data.
    Includes light augmentation to help the model generalize and reduce
    overfitting, especially important given the leukemia dataset is
    relatively small (~10K images) and imbalanced.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms():
    """
    Transform pipeline for VALIDATION/TEST data.
    No augmentation — only resize + normalize, so evaluation reflects
    the model's true performance on realistic, undistorted images.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


if __name__ == "__main__":
    # Quick sanity check — apply both transforms to one real image and confirm
    # output tensor shape/dtype are correct.
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from src.data.dataset_leukemia import LeukemiaDataset
    from src.utils.config import LEUKEMIA_RAW

    train_tf = get_train_transforms()
    eval_tf = get_eval_transforms()

    dataset = LeukemiaDataset(root_dir=LEUKEMIA_RAW, transform=train_tf)
    image, label = dataset[0]
    print(f"Train transform -> shape: {image.shape}, dtype: {image.dtype}, label: {label}")

    dataset_eval = LeukemiaDataset(root_dir=LEUKEMIA_RAW, transform=eval_tf)
    image_eval, label_eval = dataset_eval[0]
    print(f"Eval transform  -> shape: {image_eval.shape}, dtype: {image_eval.dtype}, label: {label_eval}")