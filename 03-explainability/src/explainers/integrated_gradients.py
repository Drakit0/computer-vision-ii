"""
This script contains the code for the Integrated Gradients.
"""

from typing import Literal

import torch
from torch import nn

from src.utils import normalize_tensor


class IntegratedGradients(nn.Module):
    """
    Computes Integrated Gradients for model interpretability.

    This class implements the Integrated Gradients method, which attributes the
    prediction of a model to its input features by integrating gradients along a
    straight path from a baseline to the input.

    More details can be found in: https://arxiv.org/abs/1703.01365
    """

    def __init__(self, model: nn.Module) -> None:
        """
        Constructor of the class.

        Args:
            model: Model to explain.
        """

        super().__init__()

        self.model = model

    def _initialize_baseline(
        self,
        channels: int,
        height: int,
        width: int,
        baseline: Literal["zero", "random"],
    ) -> torch.Tensor:
        """
        Initializes the baseline tensor.

        Args:
            channels: Number of channels of the image.
            height: Height of the image.
            width: Width of the image.
            baseline: Type of baseline to use.

        Returns:
            Baseline tensor of shape (channels, height, width).

        Raises:
            ValueError: If the value of 'baseline' is not valid.
        """

        match baseline:
            case "zero":
                return torch.zeros((channels, height, width))
            case "random":
                return torch.randn((channels, height, width))
            case _:
                raise ValueError("Please introduce a correct value for the baseline.")

    def forward(
        self,
        x: torch.Tensor,
        target_class: int | None = None,
        n_steps: int = 10,
        baseline: Literal["zero", "random"] = "zero",
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor. Dimensions: [batch, channels, height, width].
            target_class: Class index for which the explanation is computed. If None, it
                uses the class with the highest score for each sample.
            n_steps: Number of steps for the integration path from baseline to input.
            baseline: Type of baseline to use.

        Returns:
            Explanation. Dimensions: [batch, height, width].
        """

        B: int
        C: int
        H: int
        W: int

        B, C, H, W = x.shape

        accumulated_grads: torch.Tensor = torch.zeros(B, C, H, W, device=x.device)
        reference: torch.Tensor = torch.stack(
            [self._initialize_baseline(C, H, W, baseline) for _ in range(B)]
        ).to(x.device)

        self.model.eval()

        # Do the gradient here instead of saliency as saliency maxed channels
        for k in range(1, n_steps + 1):
            interpolation: torch.Tensor = reference + (k / n_steps) * (x - reference)
            interpolation.requires_grad_()
            output: torch.Tensor = self.model(interpolation)

            target_class_idx: torch.Tensor
            if target_class is None:
                target_class_idx = output.argmax(1)
            else:
                target_class_idx = torch.full((B,), target_class, device=x.device)

            loss: torch.Tensor = output[
                torch.arange(B, device=x.device), target_class_idx
            ].sum()
            loss.backward()

            if interpolation.grad is not None:
                accumulated_grads += interpolation.grad

            self.model.zero_grad()  # Restart grads

        integrated_grads: torch.Tensor = (x - reference) * accumulated_grads / n_steps

        attributions: torch.Tensor = integrated_grads.abs().max(1)[
            0
        ]  # Max over channels

        return normalize_tensor(attributions)
