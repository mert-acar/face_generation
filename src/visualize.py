import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from typing import List, Dict


def side_by_side(images: List[Image.Image], axis: int = 1) -> Image.Image:
    """Concatenates multiple images horizontally or vertically.

    Args:
        images: List of PIL images to combine
        axis: Concatenation axis (0 for vertical, 1 for horizontal)

    Returns:
        Combined PIL image

    Note:
        All images must have the same dimensions along the non-concatenated axis
    """
    if len(images) < 2:
        return images[0]
    combined = np.concatenate([np.array(image.convert("RGB")) for image in images], axis=axis)
    return Image.fromarray(combined)


def plot_performance_curves(metrics: Dict[str, Dict[str, List[float]]], output_path: str):
    """Plots training/validation metrics over epochs.

    Args:
        metrics: Nested dictionary of metrics in format:
                {
                    "metric_name": {
                        "train": [values...],
                        "val": [values...]
                    }
                }
        output_path: Directory to save the plot

    Saves:
        performance_curves.png: Plot of all metrics over epochs
    """
    fig, axs = plt.subplots(1, len(metrics), tight_layout=True, figsize=(5 * len(metrics), 5))
    epochs = list(range(1, len(metrics["Loss"]["train"]) + 1))
    for i, (metric, arr) in enumerate(metrics.items()):
        for phase, val in arr.items():
            axs[i].plot(epochs, val, label=phase)
        axs[i].set_xlabel("Epochs")
        axs[i].set_ylabel(metric)
        axs[i].set_title(metric)
        axs[i].legend()
        axs[i].grid(True)
    fig.suptitle("Model Performance Across Epochs")
    plt.savefig(os.path.join(output_path, "performance_curves.png"), bbox_inches="tight")
