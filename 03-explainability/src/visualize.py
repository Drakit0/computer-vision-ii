"""
Script to visualize the explanations.
"""

import os
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from torch import nn

from src.explainers.guided_backprop import GuidedBackprop
from src.explainers.integrated_gradients import IntegratedGradients
from src.explainers.occlusion import Occlusion
from src.utils import clear_folder


class Visualizer:
    """
    Class to visualize the explanations of an explainer.
    """

    def __init__(
        self,
        method: Literal["occlusion", "guided_backprop", "integrated_gradients"],
        model: nn.Module,
        normalized_image: torch.Tensor,
        unnorm_image: torch.Tensor,
    ) -> None:
        """
        Constructor of the class.

        Args:
            method: Name of the explanation method.
            model: Model we want to explain.
            normalized_image: Normalized image. Dimensions: [channels, height, width].
            unnorm_image: Unnormalized image. Dimensions: [channels, height, width].
        """

        self.explainer = self._get_explainer(method, model)
        self.normalized_image = normalized_image
        self.unnorm_image = unnorm_image
        self.explanation = torch.zeros(normalized_image.shape)

        # To generate the explanation
        self.update_images(normalized_image, unnorm_image)

    @staticmethod
    def _get_explainer(
        method: Literal["occlusion", "guided_backprop", "integrated_gradients"],
        model: nn.Module,
    ) -> nn.Module:
        """
        Obtains the desired explanation method.

        Args:
            method: Method used in the explainer.
            model: Model we want to explain.

        Returns:
            Desired explainer.

        Raises:
            ValueError: If the value of the method is not correct.
        """

        match method:
            case "occlusion":
                return Occlusion(model)
            case "guided_backprop":
                return GuidedBackprop(model)
            case "integrated_gradients":
                return IntegratedGradients(model)
            case _:
                raise ValueError("Please, introduce a correct value for the method.")

    def update_images(
        self, normalized_image: torch.Tensor, unnorm_image: torch.Tensor
    ) -> None:
        """
        Updates the normalized and unnormalized images of the class and the explanation.

        Args:
            normalized_image: Normalized image. Dimensions: [channels, height, width].
            unnorm_image: Unnormalized image. Dimensions: [channels, height, width].
        """

        self.normalized_image = normalized_image
        self.unnorm_image = unnorm_image
        self.explanation = self.explainer(self.normalized_image.unsqueeze(0))[0]

    def explain_with_plot(
        self,
        path: str = "",
    ) -> Figure:
        """
        Plots the explanation for the image.

        Args:
            path: Path to save the figure, if provided.

        Returns:
            Figure containing the explanation plot.
        """

        img = np.transpose(
            self.unnorm_image.clone().squeeze(0).detach().cpu().numpy(), (1, 2, 0)
        )

        fig, axs = plt.subplots(figsize=(14, 6))

        axs.imshow(img)
        heatmap = axs.imshow(self.explanation.squeeze(0), cmap="jet", alpha=0.5)
        axs.axis("off")
        axs.set_title(self.explainer.__class__.__name__)
        fig.colorbar(heatmap, ax=axs)

        if path is not None:
            fig.savefig(path, bbox_inches="tight")

        return fig

    def _mask_pixels(
        self, img: np.ndarray, flat_idx: np.ndarray, percentage: float, step: int
    ) -> np.ndarray:
        """
        Masks the corresponding pixels of the original image.

        Args:
            img: Original image. Dimensions: [height, width, channels].
            flat_idx: Flattened index of the top/worst pixels.
            percentage: Percentage of pixels removed in each step.
            step: Number of the current step.

        Returns:
            Copy of the original image with the pixels set to zero.
        """

        img_copy: np.ndarray = img.copy()

        if step > 0:
            H: int
            W: int

            H, W = img.shape[0], img.shape[1]
            total_pixels: int = H * W

            num_pixels_to_mask: int = int(total_pixels * percentage * step)
            pixels_to_mask: np.ndarray = flat_idx[:num_pixels_to_mask]

            row_indices: np.ndarray = pixels_to_mask // W
            col_indices: np.ndarray = pixels_to_mask % W

            img_copy[row_indices, col_indices, :] = 0

        return img_copy

    @staticmethod
    def _plot_fig(
        percentage: float,
        step: int,
        img_step: np.ndarray,
        output_folder: str,
        remove_top: bool,
    ) -> None:
        """
        Plots the figure with some removed pixels.

        Args:
            percentage: Percentage of pixels removed in each step.
            step: Number of the current step.
            img_step: Image of the current step. Dimensions: [height, width, channels].
            output_folder: Path where the figure is saved.
            remove_top: True to remove top pixels, False to remove worst pixels.
        """

        fig, axs = plt.subplots(figsize=(10, 6))
        start_title = "Top" if remove_top else "Worst"
        percentage_str = int(100 * percentage * step)

        axs.set_title(f"{start_title} {percentage_str}% pixels removed!")
        axs.imshow(img_step)
        axs.axis("off")
        if output_folder:
            fig.savefig(
                f"{output_folder}/{start_title.lower()}_{percentage_str}.png",
                bbox_inches="tight",
            )
            plt.close(fig)

    def remove_pixels(
        self,
        remove_top: bool = True,
        percentage: float = 0.2,
        save_figs: bool = False,
    ) -> None:
        """
        Removes top or worst pixels of the image according to the explanation and saves
        the figures. Original image is not modified.

        Args:
            remove_top: True to remove top pixels, False to remove worst pixels.
            percentage: Percentage of pixels removed in each step.
            save_figs: To save the generated figures or not.
        """

        output_folder = ""
        if save_figs:
            output_folder = (
                f"images/remove/{(self.explainer.__class__.__name__).lower()}/"
                f"{"top" if remove_top else "worst"}"
            )
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            else:
                clear_folder(output_folder)

        img: np.ndarray = np.transpose(
            self.unnorm_image.squeeze(0).detach().cpu().numpy(), (1, 2, 0)
        )
        explanation_flat: np.ndarray = (
            self.explanation.squeeze(0).detach().cpu().numpy().flatten()
        )

        flat_idx: np.ndarray
        if remove_top:  # Descending order
            flat_idx = np.argsort(explanation_flat)[::-1]
        else:  # Ascending order
            flat_idx = np.argsort(explanation_flat)

        num_steps: int = int(1.0 / percentage) + 1

        for step in range(num_steps):
            img_step: np.ndarray = self._mask_pixels(img, flat_idx, percentage, step)

            self._plot_fig(percentage, step, img_step, output_folder, remove_top)
