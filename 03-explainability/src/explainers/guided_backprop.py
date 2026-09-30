"""
This script contains the code for the Guided Backpropagation.
"""

from typing import Any

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from src.utils import normalize_tensor


class GuidedBackprop(nn.Module):
    """
    Computes Guided Backpropagation for model interpretability. It is IMPORTANT that the
    ReLU layers in the model are defined as torch.nn.ReLU instead of F.relu, otherwise
    hooks will not be registered correctly.

    Guided backpropagation computes the gradient of the target output with respect to
    the input, but gradients of ReLU functions are overridden so that only non-negative
    gradients are backpropagated.

    More details can be found in: https://arxiv.org/abs/1412.6806
    """

    def __init__(self, model: nn.Module) -> None:
        """
        Constructor of the class.

        Args:
            model: Model to explain.
        """

        super().__init__()

        self.model = model
        self.activation_maps: list[torch.Tensor] = []
        self.hooks: list[RemovableHandle] = []

    def _register_hooks(self) -> None:
        """
        Modifies forward and backward hooks for the ReLU and registers it in the
        modules.

        Activation maps in each layer will be saved in the forward pass and deleted in
        the backward pass. This sequential structure helps follow perfectly the forward
        and backward passes: layer_1 -> ... -> layer_n (forward),
        layer_n -> ... -> layer_1 (backward).
        """

        def backward_hook(
            module: nn.Module,
            grad_in: tuple[torch.Tensor, ...] | torch.Tensor,
            grad_out: tuple[torch.Tensor, ...] | torch.Tensor,
        ) -> tuple[torch.Tensor, ...] | torch.Tensor | None:
            """
            Backward hooks.

            Args:
                module: Module of the hook.
                gran_in: Input gradient.
                grad_out: Output gradient.

            Returns:
                Output gradient.
            """

            # Last activation map (corresponding to this layer)
            activation = self.activation_maps.pop()

            # grad_out is a tuple, get the first element
            if isinstance(grad_out, tuple):
                grad_out_tensor: torch.Tensor = grad_out[0]
            else:
                grad_out_tensor = grad_out

            # Apply guided backprop: only positive gradients
            # through positive activations
            guided_grad: torch.Tensor = (
                grad_out_tensor
                * (activation > 0).float()
                * (grad_out_tensor > 0).float()
            )

            return (guided_grad.clone(),)

        def forward_hook(
            module: nn.Module,
            inp: tuple[Any, ...],
            out: Any,
        ) -> Any | None:
            """
            Forward hooks. We will not use this, but if we wanted to extract the
            activations of each layer, for example to visualize them, we could use this.

            Args:
                module: Module of the hook.
                inp: Input tensor.
                out: Output tensor.

            Returns:
                Output tensor.
            """

            self.activation_maps.append(out.data)
            return None

        for module in self.model.modules():

            if isinstance(module, nn.ReLU):
                # Disable inplace operations to avoid autograd errors
                module.inplace = False
                self.hooks.append(module.register_forward_hook(forward_hook))
                self.hooks.append(module.register_full_backward_hook(backward_hook))

    def _delete_hooks_and_activation_maps(self) -> None:
        """
        Deletes all registered hooks.
        """

        for handle in self.hooks:
            handle.remove()

        self.hooks = []
        self.activation_maps = []

    def forward(self, x: torch.Tensor, target_class: int | None = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor. Dimensions: [batch, channels, height, width].
            target_class: Class index for which the explanation is computed. If None, it
                uses the class with the highest score for each sample.

        Returns:
            Explanation. Dimensions: [batch, height, width].

        Raises:
            RuntimeError: If no ReLU modules are detected.
        """

        self.model.eval()
        x.requires_grad_()

        self._register_hooks()

        if len(self.hooks) == 0:
            raise RuntimeError("no ReLU modules found")

        B: int = x.shape[0]

        output: torch.Tensor = self.model(x)

        target_class_idx: torch.Tensor
        if target_class is None:
            target_class_idx = output.argmax(1)
        else:
            target_class_idx = torch.full((B,), target_class)

        loss: torch.Tensor = output[torch.arange(B), target_class_idx].sum()
        loss.backward()

        self._delete_hooks_and_activation_maps()

        if x.grad is None:
            raise RuntimeError("Gradient is None")

        grad: torch.Tensor = x.grad.abs()

        return normalize_tensor(grad.max(1)[0])
