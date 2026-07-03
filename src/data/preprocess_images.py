from torchvision import transforms
from src.utils.config import IMAGE_SIZE

# ImageNet normalization stats — required since we're using a pretrained ResNet18.
# The pretrained weights expect inputs normalized this exact way.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_train_transforms():
    """
    Transform pipeline for training data.
    Includes light augmentation to reduce overfitting — cell images can be
    flipped/rotated freely since orientation carries no diagnostic meaning.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_eval_transforms():
    """
    Transform pipeline for validation/test data.
    No augmentation — we want consistent, reproducible evaluation.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])