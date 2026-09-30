import torch
import torch.nn.functional as F
from torch import nn


class Residual(nn.Module):
    """Residual block with skip connection.

    A residual block applies a sequence of operations and adds the result
    to the input (skip connection). This helps with gradient flow in deep networks.
    """

    def __init__(
        self, in_channels: int, num_hiddens: int, num_residual_hiddens: int
    ) -> None:
        """Initialize residual block.

        Args:
            in_channels: Number of input channels.
            num_hiddens: Number of output channels (should equal in_channels for residual).
            num_residual_hiddens: Number of hidden channels in the block.
        """
        super().__init__()
        self._block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_residual_hiddens,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=num_residual_hiddens,
                out_channels=num_hiddens,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor.

        Returns:
            Output tensor with residual connection applied.
        """
        return x + self._block(x)


class ResidualStack(nn.Module):
    """Stack of residual blocks."""

    def __init__(
        self,
        in_channels: int,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
    ) -> None:
        """Initialize stack of residual blocks.

        Args:
            in_channels: Number of input channels.
            num_hiddens: Number of hidden channels.
            num_residual_layers: Number of residual blocks to stack.
            num_residual_hiddens: Hidden channels in each residual block.
        """
        super().__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList(
            [
                Residual(in_channels, num_hiddens, num_residual_hiddens)
                for _ in range(num_residual_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual stack.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after passing through all residual blocks and final ReLU.
        """
        for layer in self._layers:
            x = layer(x)
        return F.relu(x)


class Encoder(nn.Module):
    """VQ-VAE encoder with convolutional layers and residual blocks.

    The encoder progressively downsamples the input image while increasing
    the number of channels, ending with residual blocks for feature refinement.
    """

    def __init__(
        self,
        in_channels: int,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
    ) -> None:
        """Initialize encoder.

        Args:
            in_channels: Number of input channels (typically 3 for RGB).
            num_hiddens: Number of hidden channels.
            num_residual_layers: Number of residual layers.
            num_residual_hiddens: Hidden channels in residual blocks.
        """
        super().__init__()

        # First conv: downsample by 2
        self._conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        # Second conv: downsample by 2 again
        self._conv_2 = nn.Conv2d(
            in_channels=num_hiddens // 2,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        # Third conv: maintain spatial dims
        self._conv_3 = nn.Conv2d(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Residual stack for feature refinement
        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation.

        Args:
            inputs: Input images [B, C, H, W].

        Returns:
            Encoded latent representation [B, num_hiddens, H//4, W//4].
        """
        x = self._conv_1(inputs)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)

        return self._residual_stack(x)


class Decoder(nn.Module):
    """VQ-VAE decoder with transposed convolutions and residual blocks.

    The decoder upsamples the quantized latent representation back to
    the original image resolution.
    """

    def __init__(
        self,
        in_channels: int,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
    ) -> None:
        """Initialize decoder.

        Args:
            in_channels: Number of input channels (embedding_dim).
            num_hiddens: Number of hidden channels.
            num_residual_layers: Number of residual layers.
            num_residual_hiddens: Hidden channels in residual blocks.
        """
        super().__init__()

        # Initial conv to project from embedding dim to hidden channels
        self._conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Residual stack for feature refinement
        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )

        # Transposed conv: upsample by 2
        self._conv_trans_1 = nn.ConvTranspose2d(
            in_channels=num_hiddens,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        # Transposed conv: upsample by 2 again, output 3 channels (RGB)
        self._conv_trans_2 = nn.ConvTranspose2d(
            in_channels=num_hiddens // 2,
            out_channels=3,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Decode quantized representation to reconstruction.

        Args:
            inputs: Quantized latent representation [B, embedding_dim, H, W].

        Returns:
            Reconstructed images [B, 3, H*4, W*4].
        """
        x = self._conv_1(inputs)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)

        x = self._conv_trans_2(x)

        return x