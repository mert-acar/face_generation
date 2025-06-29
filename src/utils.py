import os
import re
import torch
import random
from PIL import Image
from shutil import rmtree
from torchvision.transforms.v2.functional import to_pil_image

from typing import List, Tuple


def create_dir(path: str):
    """Creates a directory, prompting for deletion if it already exists.

    Args:
        path: Directory path to create

    Note:
        Will prompt user for confirmation before deleting existing directory
    """
    if os.path.exists(path):
        c = input(f"Output path {path} is not empty! Do you want to delete the folder [y / n]: ")
        if "y" == c.lower():
            rmtree(path, ignore_errors=True)
        else:
            print("Exit!")
            return
    os.makedirs(path)


def post_process(input: torch.Tensor, denormalize: bool = True) -> List[Image.Image]:
    """Converts model output tensors to PIL images.

    Args:
        input: Batch of image tensors
        denormalize: Whether to denormalize using ImageNet stats

    Returns:
        List of PIL images
    """
    if denormalize:
        mean = torch.tensor([0.485, 0.456, 0.406], device=input.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=input.device).view(1, 3, 1, 1)
        input = (input * std) + mean
    return [to_pil_image(x) for x in input]


def train_test_split(datalist: List, train_size: float = 0.85) -> Tuple[List, List]:
    """Randomly splits a dataset into train and test sets.

    Args:
        datalist: List of data items to split
        train_size: Fraction of data to use for training

    Returns:
        (train_list, test_list) tuple
    """
    num_pairs = len(datalist)
    random.shuffle(datalist)
    train = datalist[: int(num_pairs * train_size)]
    test = datalist[int(num_pairs * train_size) :]
    return train, test


def get_image_pairs(folder_path: str) -> List[Tuple[str, str]]:
    """Finds matching clean/bearded image pairs in a directory.

    Matches files named 'clean_X.png' with corresponding 'bearded_X.png'
    where X is a numeric identifier.

    Args:
        folder_path: Directory containing the image pairs

    Returns:
        List of (clean_filename, bearded_filename) tuples
    """
    files = os.listdir(folder_path)
    clean_pattern = re.compile(r"clean_(\d+)\.png")
    beard_pattern = re.compile(r"bearded_(\d+)\.png")
    clean_files, beard_files = {}, {}
    for file in files:
        clean_match = clean_pattern.match(file)
        beard_match = beard_pattern.match(file)

        if clean_match:
            idx = int(clean_match.group(1))
            clean_files[idx] = file
        elif beard_match:
            idx = int(beard_match.group(1))
            beard_files[idx] = file

    pairs = [(clean_files[idx], beard_files[idx]) for idx in clean_files if idx in beard_files]
    return pairs
