"""
This script contains the code for the Occlusion.
"""

import torch
from torch import nn

from src.utils import normalize_tensor


class Occlusion(nn.Module):
    """
    Computes Occlusion for model interpretability.

    This class provides methods to generate feature or pixel-level importance
    maps by systematically occluding parts of the input and measuring the
    impact on the model's output.

    More details can be found in: https://arxiv.org/abs/1311.2901
    """

    def __init__(self, model: nn.Module) -> None:
        """
        Constructor of the class.

        Args:
            model: Model to explain.
        """

        super().__init__()

        self.model = model

    def _get_target_outputs(
        self, original_output: torch.Tensor, target_class: int | None
    ):
        """
        Gets the output values and target indices for the specified class.

        Args:
            original_output: The model's output tensor.
            target_class: Class index for which the explanation is computed. If None, it
                uses the class with the highest score for each sample.

        Returns:
            Output values for the target class and target class indices as a tensor.

        Raises:
            ValueError: If target_class is not None or an integer.
        """

        if target_class is not None and not isinstance(target_class, int):
            raise ValueError("target_class must be a None or an integer")

        target_class_indexes: torch.Tensor
        if target_class is None:
            target_class_indexes = original_output.argmax(-1, True)
        else:
            target_class_indexes = torch.full(
                (original_output.shape[0], 1), target_class
            )

        values: torch.Tensor = torch.take_along_dim(
            original_output, target_class_indexes, -1
        ).squeeze(-1)

        return values, target_class_indexes

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        target_class: int | None = None,
        mask_size: int | tuple[int, int] | None = None,
        stride: int | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch, channels, height, width].
            target_class: Class index for which the explanation is computed. If None, it
                uses the class with the highest score for each sample.
            mask_size: Shape of the occlusion mask. If int, creates a square mask. If
                tuple, uses (width, height). If None, it uses a mask ten times smaller
                than the width and height.
            stride: Stride for moving the occlusion mask. If None, it uses half the mask
                size.

        Returns:
            Explanation. Dimensions: [batch, height, width].
        """

        B: int
        H: int
        W: int
        B, _, H, W = x.shape

        seen_pixels: torch.Tensor = torch.zeros((B, H, W))
        occlussion_values: torch.Tensor = torch.zeros((B, H, W))

        # Set the w and h of the masks
        w_mask: int
        h_mask: int
        if isinstance(mask_size, int):
            w_mask = mask_size
            h_mask = mask_size
        elif isinstance(mask_size, tuple):
            w_mask = mask_size[0]
            h_mask = mask_size[1]
        else:
            w_mask = max(1, int(W / 10))
            h_mask = max(1, int(H / 10))

        # Mask can't be bigger than the dimension over the one it will be applied
        w_mask = min(W, w_mask)
        h_mask = min(H, h_mask)

        stride_w: int
        stride_h: int
        if stride is None:
            stride_w = max(1, int(w_mask / 2))
            stride_h = max(1, int(h_mask / 2))
        else:
            stride_w = stride
            stride_h = stride

        self.model.eval()

        # No need to execute always
        result: torch.Tensor = self.model(x)

        target_class_indexes: torch.Tensor
        if target_class is None:
            target_class_indexes = result.argmax(1, True)
        else:
            target_class_indexes = torch.full((B, 1), target_class)

        result_target: torch.Tensor = torch.take_along_dim(result, target_class_indexes)

        i: int = 0
        j: int = 0

        completed_occlusion: bool = False

        while not completed_occlusion:
            x0 = i
            y0 = j
            x1 = min(x0 + w_mask, W)
            y1 = min(y0 + h_mask, H)

            x_copy: torch.Tensor = x.clone().detach()
            x_copy[..., y0:y1, x0:x1] = 0

            result_occlussion: torch.Tensor = self.model(x_copy)
            result_occlussion_target: torch.Tensor = torch.take_along_dim(
                result_occlussion, target_class_indexes
            )
            diff: torch.Tensor = torch.abs(result_target - result_occlussion_target)

            seen_pixels[:, y0:y1, x0:x1] += 1
            occlussion_values[:, y0:y1, x0:x1] += diff[:, None, None]

            if i + stride_w <= W - w_mask:
                i += stride_w

            elif i != W - w_mask:
                i = W - w_mask

            else:  # Next row
                i = 0

                if j + stride_h <= H - h_mask:
                    j += stride_h

                elif j != H - h_mask:
                    j = H - h_mask

                else:  # Bottom right
                    completed_occlusion = True

        occlussion_result: torch.Tensor = torch.div(occlussion_values, seen_pixels)

        # Replaces the nans that appear because of the jumping stride
        # that leaves 0s in seen_pixels
        occlussion_result_no_nans: torch.Tensor = torch.nan_to_num(occlussion_result)

        return normalize_tensor(occlussion_result_no_nans)
