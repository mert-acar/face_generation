# Beard Generation and Image-to-Image Translation with Deep Learning

**Author**: Mert Acar

This project implements a deep learning solution for removing beards on facial images. The implementation includes various model architectures, training pipelines, and evaluation metrics. This project is the submission for the case study.

## Project Structure

```
.
├── src/
│   ├── dataset.py        # Dataset and data loading utilities
│   ├── generate.py       # Beard mask generation and inference pipelines
│   ├── loss.py           # Loss functions and loss registry
│   ├── model.py          # Model architectures and loading utilities
│   ├── test.py           # Testing and evaluation metrics
│   ├── train.py          # Training pipeline
│   ├── utils.py          # General utility functions
│   └── visualize.py      # Visualization utilities
└── AppNation_case_study.ipynb  # Main notebook with examples and usage
```

## Features

- Multiple model architectures (UNet, UNet++, DeepLabV3+, etc.)
- Custom beard region mask generation using MediaPipe face detection
- Multiple loss functions including perceptual loss (LPIPS)
- Comprehensive evaluation metrics (SSIM, PSNR, L1, Perceptual)
- Training pipeline with early stopping and learning rate scheduling
- Visualization tools for model performance and results

## Requirements

- Python 3.11+
- PyTorch
- torchvision
- albumentations
- MediaPipe
- scikit-image
- matplotlib
- PIL
- tqdm
- lpips

## Getting Started

1. Install the required packages:
```bash
uv pip sync pyproject.toml
```

2. Open and run the `case_study.ipynb` notebook for examples and usage.

## Usage

The main usage examples and workflows are documented in `case_study.ipynb`. The notebook includes:
- Data preparation and loading
- Model configuration and training
- Inference and evaluation
- Result visualization
