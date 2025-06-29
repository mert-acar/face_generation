import os
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from typing import Tuple, Optional, Dict, Any


def build_transforms(transforms: Optional[Dict[str, Dict[str, Any]]] = None) -> A.Compose:
    """Builds an albumentations transform pipeline with normalization.

    Args:
        transforms: Dictionary mapping transform names to their parameters.
                   Format: {'TransformName': {'param1': value1, ...}}

    Returns:
        Composed transformation pipeline with normalization and tensor conversion.
    """
    transform_list = [A.Normalize(normalization="min_max"), ToTensorV2()]
    augmentations = []
    if transforms is not None:
        for transform, kwargs in transforms.items():
            try:
                trans = getattr(A, transform)
            except AttributeError:
                print(f"Transform {transform} cannot be found in albumentations")
                continue
            augmentations.append(trans(**kwargs))
    return A.Compose(augmentations + transform_list, additional_targets={"target": "image"})


def create_dataloader(
    data_path: str,
    transform: Optional[Dict[str, Dict[str, Any]]] = None,
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle=False,
) -> DataLoader:
    """Creates a DataLoader for the BeardDataset."""
    return DataLoader(
        BeardDataset(data_path, transform),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
    )


class BeardDataset(Dataset):
    """Dataset for paired beard/clean face images.

    Expects data directory to contain pairs of images:
        - *clean.png: Face without beard
        - *bearded.png: Corresponding face with beard
    """

    def __init__(self, data_path: str, transform: Optional[Dict[str, Dict[str, Any]]] = None):
        self.data_path = data_path
        fnames = [
            fname
            for fname in os.listdir(self.data_path)
            if (fname.endswith(".png") and "clean" in fname)
        ]
        self.data_list = [(fname, fname.replace("clean", "bearded")) for fname in fnames]
        self.transform = build_transforms(transform)

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        beard_image = np.array(Image.open(os.path.join(self.data_path, self.data_list[idx][1])))
        clean_image = np.array(Image.open(os.path.join(self.data_path, self.data_list[idx][0])))
        if self.transform is not None:
            transformed = self.transform(image=beard_image, target=clean_image)
            beard_image = transformed["image"]
            clean_image = transformed["target"]
        return beard_image, clean_image
