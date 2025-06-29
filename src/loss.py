import lpips
import torch
from typing import Callable, Any, Dict


class LossRegistry:
    """Registry for managing and instantiating loss functions."""

    def __init__(self):
        self.loss_classes = {
            "l1_loss": torch.nn.L1Loss,
            "mse_loss": torch.nn.MSELoss,
        }

    def register(self, name: str):
        """Decorator to register new loss functions.

        Args:
            name: Identifier for the loss function
        """

        def decorator(cls: Callable):
            self.loss_classes[name] = cls
            return cls

        return decorator

    def get(self, name: str, **kwargs):
        """Instantiates a registered loss function.

        Args:
            name: Identifier of the loss function
            **kwargs: Arguments passed to the loss constructor

        Raises:
            ValueError: If loss name is not found in registry
        """
        if name in self.loss_classes:
            return self.loss_classes[name](**kwargs)
        else:
            raise ValueError(f"Loss class '{name}' not found in the registry.")


loss_registry = LossRegistry()


@loss_registry.register("perceptual_loss")
class PerceptualLoss(torch.nn.Module):
    """LPIPS perceptual loss using pretrained networks."""

    def __init__(self, perceptual_net: str = "vgg"):
        """
        Args:
            perceptual_net: Backend network architecture ('vgg' or 'alex')
        """
        super(PerceptualLoss, self).__init__()
        self.perceptual_loss = lpips.LPIPS(net=perceptual_net).eval()

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.perceptual_loss(generated, target).mean()


class BeardLoss(torch.nn.Module):
    """Combined loss function for beard generation.

    Combines multiple loss functions with configurable weights.

    Args:
        modules: Dictionary of loss configurations:
                {
                    "loss_name": {
                        "weight": float,
                        "args": dict of constructor arguments
                    }
                }
    """

    def __init__(self, modules: Dict[str, Dict[str, Any]]):
        super().__init__()
        self.losses = torch.nn.ModuleDict()
        self.weights = {}
        for loss_name, args in modules.items():
            kwargs = args.get("args", {})
            self.losses[loss_name] = loss_registry.get(loss_name, **kwargs)
            weight = args.get("weight", 1)
            self.weights[loss_name] = weight

    def forward(self, generated: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes weighted sum of all registered losses."""
        total_loss = 0
        for key in self.losses:
            total_loss += self.losses[key](generated, targets) * self.weights[key]
        return total_loss


if __name__ == "__main__":
    args = {
        "mse_loss": {"weight": 1},
        "perceptual_loss": {"weight": 1, "perceptual_net": "vgg"},
    }
    loss = BeardLoss(args)
