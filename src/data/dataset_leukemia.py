"""
PyTorch Dataset for the C-NMC 2019 Leukemia dataset.
Loads cell images from data/raw/leukemia/all (cancer) and data/raw/leukemia/hem (healthy).
"""

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class LeukemiaDataset(Dataset):
    """
    Binary classification dataset: cancer (all) vs healthy (hem).
    Label convention: 1 = cancer (ALL), 0 = healthy (hem)
    """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: path to data/raw/leukemia (must contain 'all/' and 'hem/' subfolders)
            transform: torchvision transforms to apply to each image
        """
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.samples = []  # list of (filepath, label)

        cancer_dir = self.root_dir / "all"
        healthy_dir = self.root_dir / "hem"

        for img_path in sorted(cancer_dir.glob("*.bmp")):
            self.samples.append((img_path, 1))

        for img_path in sorted(healthy_dir.glob("*.bmp")):
            self.samples.append((img_path, 0))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found in {cancer_dir} or {healthy_dir}. "
                "Check that the folders exist and contain .bmp files."
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
    from src.utils.config import LEUKEMIA_RAW

    dataset = LeukemiaDataset(root_dir=LEUKEMIA_RAW)
    print(f"Total samples: {len(dataset)}")
    print(f"Cancer (label=1): {sum(1 for _, l in dataset.samples if l == 1)}")
    print(f"Healthy (label=0): {sum(1 for _, l in dataset.samples if l == 0)}")

    image, label = dataset[0]
    print(f"First sample -> size: {image.size}, label: {label}")