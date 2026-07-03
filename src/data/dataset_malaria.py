"""
PyTorch Dataset for the NIH Malaria Cell Images dataset.
Loads cell images from data/raw/malaria/Parasitized and data/raw/malaria/Uninfected.
"""

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class MalariaDataset(Dataset):
    """
    Binary classification dataset: parasitized vs uninfected.
    Label convention: 1 = parasitized (infected), 0 = uninfected (healthy)
    """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: path to data/raw/malaria (must contain 'Parasitized/' and 'Uninfected/' subfolders)
            transform: torchvision transforms to apply to each image
        """
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.samples = []  # list of (filepath, label)

        parasitized_dir = self.root_dir / "Parasitized"
        uninfected_dir = self.root_dir / "Uninfected"

        for img_path in sorted(parasitized_dir.glob("*.png")):
            self.samples.append((img_path, 1))

        for img_path in sorted(uninfected_dir.glob("*.png")):
            self.samples.append((img_path, 0))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found in {parasitized_dir} or {uninfected_dir}. "
                "Check that the folders exist and contain .png files."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == "__main__":
    # Quick manual test — run this file directly to sanity-check the dataset loads correctly
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from src.utils.config import MALARIA_RAW

    dataset = MalariaDataset(root_dir=MALARIA_RAW)
    print(f"Total samples: {len(dataset)}")
    print(f"Parasitized (label=1): {sum(1 for _, l in dataset.samples if l == 1)}")
    print(f"Uninfected (label=0): {sum(1 for _, l in dataset.samples if l == 0)}")

    image, label = dataset[0]
    print(f"First sample -> size: {image.size}, label: {label}")

    # Check a few more sizes since NIH malaria images are known to vary in dimensions
    sizes = set()
    for i in range(0, len(dataset), len(dataset) // 20):  # sample ~20 images spread across dataset
        img, _ = dataset[i]
        sizes.add(img.size)
    print(f"Sample of image sizes found: {sizes}")