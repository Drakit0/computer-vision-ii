"""
Script to define some utility functions.
"""

import os

import torch
from torch import nn


def saliency_map(
    model: nn.Module, x: torch.Tensor, target_class: int | None
) -> torch.Tensor:
    """
    Computes gradients of the model output with respect to input. As we want to return a
    saliency map, we must obtain the maximum of the absolute values along the channels
    dimension.

    Args:
        model: Model we want to explain.
        x: Input tensor. Dimensions: [batch, channels, height, width].
        target_class: Class for which backward pass is computed. If None, then we assume
            it is the one with maximum score.

    Returns:
        Saliency map of the model output with respect to input. Dimensions: [batch,
        height, width].

    Raises:
        RuntimeError: If the gradients calculated with the backward are None.
    """

    model.eval()
    x.requires_grad_()
    output: torch.Tensor = model(x)

    target_class_idx: torch.Tensor
    if target_class is None:
        target_class_idx = output.argmax(1)
    else:
        target_class_idx = torch.full((x.shape[0],), target_class)

    loss: torch.Tensor = output[torch.arange(x.shape[0]), target_class_idx].sum()
    loss.backward()

    if x.grad is None:
        raise RuntimeError("Gradient is None")

    grad: torch.Tensor = x.grad.abs()

    saliency: torch.Tensor = grad.max(dim=1)[0]  # Max values (not indexes) of channels

    return saliency


def normalize_tensor(explanation: torch.Tensor) -> torch.Tensor:
    """
    Normalizes the explanation tensor.

    Args:
        explanation: Explanation tensor with dimensions [batch, height, width].

    Returns:
        Normalized explanation with the same dimensions.
    """

    max_ = torch.amax(explanation, dim=(1, 2), keepdim=True)
    min_ = torch.amin(explanation, dim=(1, 2), keepdim=True)

    return (explanation - min_) / (max_ - min_ + 1e-8)


def clear_folder(folder_path: str, extension: str = ".png") -> None:
    """
    Clears all the images that match an extension in a folder.

    Ags:
        folder_path: Path of the folder.
        extension: Extension of the files we want to remove.
    """

    for file in os.listdir(folder_path):
        if file.endswith(extension):
            os.remove(os.path.join(folder_path, file))
