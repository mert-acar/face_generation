import os
import json
import torch
import segmentation_models_pytorch as smp
from typing import Union, Tuple, Any, Dict


def get_model(model_name: str, **kwargs) -> torch.nn.Module:
    """Factory function for creating models.

    Args:
        model_name: Name of the model architecture
        **kwargs: Model-specific parameters

    Returns:
        Instantiated model

    Raises:
        ValueError: If model_name is not recognized
    """
    model_name = model_name.lower()
    model_mapping = {
        "unet": smp.Unet,
        "unetplusplus": smp.UnetPlusPlus,
        "unet++": smp.UnetPlusPlus,
        "manet": smp.MAnet,
        "linknet": smp.Linknet,
        "fpn": smp.FPN,
        "pspnet": smp.PSPNet,
        "pan": smp.PAN,
        "deeplabv3": smp.DeepLabV3,
        "deeplabv3plus": smp.DeepLabV3Plus,
        "segformer": smp.Segformer,
    }

    if model_name not in model_mapping:
        raise ValueError(
            f"Unknown model: {model_name}. Available models: {', '.join(model_mapping.keys())}"
        )

    default_kwargs = {
        "encoder_name": "resnet34",
        "encoder_weights": "imagenet",
        "in_channels": 3,
        "classes": 3,
    }
    default_kwargs.update(kwargs)
    return model_mapping[model_name](**default_kwargs)


def load_model(
    experiment_path: str, load_config: bool = True
) -> Union[torch.nn.Module, Tuple[torch.nn.Module, Dict[Any, Any]]]:
    """Loads a model and optionally its configuration from an experiment directory.

    Args:
        experiment_path: Path to the experiment directory
        load_config: Whether to load and return the configuration

    Returns:
        If load_config is True, returns (model, config)
        If load_config is False, returns only model
    """
    device = torch.device("cpu")
    with open(os.path.join(experiment_path, "ExperimentSummary.json"), "r") as f:
        config = json.load(f)
    model = get_model(**config["model"])
    state_dict = torch.load(
        os.path.join(experiment_path, "checkpoint.pt"), weights_only=True, map_location=device
    )
    model.load_state_dict(state_dict)

    if load_config:
        return model, config
    return model
