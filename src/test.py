import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Any
from loss import PerceptualLoss
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from typing import List, Dict

from dataset import create_dataloader


def calculate_ssim(pred: np.ndarray, original: np.ndarray) -> float:
    """Calculates mean SSIM score across a batch of images.

    Args:
        pred: Predicted images (N, C, H, W)
        original: Ground truth images (N, C, H, W)

    Returns:
        Mean SSIM score across batch

    Raises:
        AssertionError: If pred and original shapes don't match
    """
    assert (
        pred.shape == original.shape
    ), f"prediction and original images do not have the same shape: {pred.shape}, {original.shape}"
    score = 0
    for i in range(len(pred)):
        score += ssim(pred[i], original[i], channel_axis=0, data_range=1)
    if len(pred) > 0:
        score /= len(pred)
    return score


def calculate_psnr(pred: np.ndarray, original: np.ndarray) -> float:
    """Calculates mean PSNR score across a batch of images.

    Args:
        pred: Predicted images (N, C, H, W)
        original: Ground truth images (N, C, H, W)

    Returns:
        Mean PSNR score across batch

    Raises:
        AssertionError: If pred and original shapes don't match
    """
    assert (
        pred.shape == original.shape
    ), f"prediction and original images do not have the same shape: {pred.shape}, {original.shape}"
    score = 0
    for i in range(len(pred)):
        score += psnr(original[i], pred[i], data_range=1)
    if len(pred) > 0:
        score /= len(pred)
    return score


PERCEPTUAL = None


def get_metric_scores(
    metric_list: List[str], generated: torch.Tensor, original: torch.Tensor
) -> Dict[str, float]:
    """Calculates multiple image quality metrics between generated and original images.

    Supported metrics:
        - SSIM: Structural similarity
        - PSNR: Peak signal-to-noise ratio
        - L1Loss: Mean absolute error
        - Perceptual: LPIPS perceptual similarity

    Args:
        metric_list: List of metric names to calculate
        generated: Generated images
        original: Ground truth images

    Returns:
        Dictionary of metric names to scores
    """
    out = {}
    for metric in metric_list:
        if metric.lower() == "ssim":
            out[metric] = calculate_ssim(
                generated.detach().cpu().numpy(), original.detach().cpu().numpy()
            )
        elif metric.lower() == "psnr":
            out[metric] = calculate_psnr(
                generated.detach().cpu().numpy(), original.detach().cpu().numpy()
            )
        elif metric.lower() == "l1loss":
            with torch.no_grad():
                out[metric] = torch.nn.functional.l1_loss(generated, original).item()
        elif metric.lower() == "perceptual":
            global PERCEPTUAL
            if PERCEPTUAL is None:
                PERCEPTUAL = PerceptualLoss().to(generated.device)
            with torch.no_grad():
                out[metric] = PERCEPTUAL(generated, original).item()
        else:
            print(f"Metric [{metric}] is not implemented, skipping...")
    return out


def test_model(model: torch.nn.Module, config: Dict[Any, Any]):
    """Evaluates model performance on test dataset using multiple metrics.

    Args:
        model: Model to evaluate
        config: Configuration dictionary containing:
               - data.data_path: Path to data directory
               - training.dataloader_args: DataLoader configuration

    Returns:
        Dictionary containing mean and standard deviation for each metric
    """
    device = torch.device(next(model.parameters()).device)

    dataloader = create_dataloader(
        data_path=os.path.join(config["data"]["data_path"], "test"),
        shuffle=False,
        **config["training"]["dataloader_args"],
    )

    model.eval()

    metrics = {"L1Loss": [], "SSIM": [], "PSNR": [], "Perceptual": []}

    with torch.no_grad():
        for data, target in tqdm(dataloader, total=len(dataloader), ncols=94):
            data, target = data.to(device), target.to(device)
            output = model(data)
            scores = get_metric_scores(metrics, output, target)
            for key, score in scores.items():
                metrics[key].append(score)

    metrics_summary = {}
    for key in metrics:
        metrics_summary[f"{key.lower()}_mean"] = np.mean(metrics[key])
        metrics_summary[f"{key.lower()}_std"] = np.std(metrics[key])
    return metrics_summary
