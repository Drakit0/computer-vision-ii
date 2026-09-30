"""
Script to create the Discriminator and Generator for the GAN.

The Discriminator takes a 1x28x28 image and applies convolutions to output a single
probability score indicating whether the image is real or fake.

The Generator takes a noise vector from the latent space and uses transposed
convolutions to generate a 1x28x28 image.
"""

import torch
from torch import nn
import torch.nn.functional as F


class DiscBlock(nn.Module):
    """
    Class to build a discriminator block consisting of Conv + BatchNorm + LeakyReLU.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        use_batchnorm: bool = True,
    ) -> None:
        """
        Constructor of the class.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            kernel_size: Kernel size.
            stride: Stride.
            padding: Padding.
            use_batchnorm: Whether to use batch normalization.
        """

        super().__init__()

        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=not use_batchnorm,
            )
        ]

        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor. Dimensions: [batch, in_channels, height, width].

        Returns:
            Output tensor. Dimensions: [batch, out_channels, h_out, w_out].
        """

        return self.block(x)


class GenBlock(nn.Module):
    """
    Class to build a generator block consisting of ConvTranspose + BatchNorm + ReLU.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        use_batchnorm: bool = True,
    ) -> None:
        """
        Constructor of the class.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            kernel_size: Kernel size.
            stride: Stride.
            padding: Padding.
            use_batchnorm: Whether to use batch normalization.
        """

        super().__init__()

        layers: list[nn.Module] = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=not use_batchnorm,
            )
        ]

        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor. Dimensions: [batch, in_channels, height, width].

        Returns:
            Output tensor. Dimensions: [batch, out_channels, h_out, w_out].
        """

        return self.block(x)


class Discriminator(nn.Module):
    """
    Discriminator network for DCGAN. Takes 1x28x28 images and outputs a probability
    score indicating whether the image is real or fake.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 32,
    ) -> None:
        """
        Constructor of the class.

        Args:
            in_channels: Number of input channels (1 for grayscale MNIST).
            out_channels: Base number of output channels for the first conv layer.
        """

        super().__init__()

        self.disc = nn.Sequential(
            # Input: [batch, 1, 28, 28]
            DiscBlock(in_channels, out_channels, 4, 2, 1, use_batchnorm=False),
            # [batch, 32, 14, 14]
            DiscBlock(out_channels, out_channels * 2, 4, 2, 1),
            # [batch, 64, 7, 7]
            DiscBlock(out_channels * 2, out_channels * 4, 3, 2, 1),
            # [batch, 128, 4, 4]
            nn.Flatten(),
            nn.Linear(out_channels * 4 * 4 * 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor. Dimensions: [batch, 1, 28, 28].

        Returns:
            Output tensor with logits (not sigmoid). Dimensions: [batch, 1].
        """

        return self.disc(x)


class Generator(nn.Module):
    """
    Generator network for DCGAN. Takes a noise vector from the latent space and
    generates 1x28x28 images.
    """

    def __init__(
        self,
        latent_dim: int = 100,
        out_channels: int = 32,
    ) -> None:
        """
        Constructor of the class.

        Args:
            latent_dim: Dimension of the latent noise vector.
            out_channels: Base number of output channels.
        """

        super().__init__()

        self.latent_dim = latent_dim

        # Project and reshape noise vector to initial feature map
        self.project = nn.Sequential(
            nn.Linear(latent_dim, out_channels * 4 * 7 * 7),
            nn.BatchNorm1d(out_channels * 4 * 7 * 7),
            nn.ReLU(inplace=True),
        )

        self.gen = nn.Sequential(
            # Input: [batch, 128, 7, 7]
            GenBlock(out_channels * 4, out_channels * 2, 4, 2, 1),
            # [batch, 64, 14, 14]
            GenBlock(out_channels * 2, out_channels, 4, 2, 1),
            # [batch, 32, 28, 28]
            nn.ConvTranspose2d(out_channels, 1, 3, 1, 1),
            # [batch, 1, 28, 28]
            nn.Sigmoid(),  # Output in range [0, 1]
        )

        self._out_channels = out_channels

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            z: Noise tensor from latent space. Dimensions: [batch, latent_dim].

        Returns:
            Generated images. Dimensions: [batch, 1, 28, 28].
        """

        # Project and reshape
        x: torch.Tensor = self.project(z)
        x = x.view(-1, self._out_channels * 4, 7, 7)

        return self.gen(x)


def gan_loss(
    disc_output: torch.Tensor,
    real_images: bool,
) -> torch.Tensor:
    """
    Computes the GAN loss for the discriminator or generator.

    The loss uses Binary Cross Entropy with logits. For real images, the target is 1
    (the discriminator should output high values). For fake images, the target is 0
    (the discriminator should output low values).

    Args:
        disc_output: Output logits from the discriminator. Dimensions: [batch, 1].
        real_images: If True, compute loss for real images (target=1).
                     If False, compute loss for fake images (target=0).

    Returns:
        The computed BCE loss.
    """

    batch_size: int = disc_output.shape[0]
    device: torch.device = disc_output.device

    if real_images:
        targets: torch.Tensor = torch.ones(batch_size, device=device)
    else:
        targets = torch.zeros(batch_size, device=device)

    loss: torch.Tensor = F.binary_cross_entropy_with_logits(
        disc_output.squeeze(), targets
    )

    return loss


def get_disc_loss(
    gen: nn.Module,
    disc: nn.Module,
    criterion: nn.Module,
    real: torch.Tensor,
    num_images: int,
    z_dim: int,
    device: str,
) -> torch.Tensor:
    """
    Computes the discriminator loss on both real and fake images.

    The discriminator tries to correctly classify real images as real (1) and
    fake images as fake (0).

    Args:
        gen: Generator network.
        disc: Discriminator network.
        criterion: Loss criterion (e.g., BCEWithLogitsLoss).
        real: Real images tensor. Dimensions: [batch, channels, height, width].
        num_images: Number of images in the batch.
        z_dim: Dimension of the latent noise vector.
        device: Device to use ('cpu' or 'cuda').

    Returns:
        The discriminator loss (average of real and fake losses).
    """

    from src.utils import get_noise

    # Loss on real images
    real_output: torch.Tensor = disc(real)
    real_target: torch.Tensor = torch.ones(num_images, 1, device=device)
    real_loss: torch.Tensor = criterion(real_output, real_target)

    # Generate fake images
    noise: torch.Tensor = get_noise(num_images, z_dim, device)
    fake_images: torch.Tensor = gen(noise).detach()  # Detach to avoid updating gen

    # Loss on fake images
    fake_output: torch.Tensor = disc(fake_images)
    fake_target: torch.Tensor = torch.zeros(num_images, 1, device=device)
    fake_loss: torch.Tensor = criterion(fake_output, fake_target)

    # Total discriminator loss
    disc_loss: torch.Tensor = (real_loss + fake_loss) / 2

    return disc_loss


def get_gen_loss(
    gen: nn.Module,
    disc: nn.Module,
    criterion: nn.Module,
    num_images: int,
    z_dim: int,
    device: str,
) -> torch.Tensor:
    """
    Computes the generator loss.

    The generator tries to fool the discriminator into classifying fake images
    as real (target=1).

    Args:
        gen: Generator network.
        disc: Discriminator network.
        criterion: Loss criterion (e.g., BCEWithLogitsLoss).
        num_images: Number of images to generate.
        z_dim: Dimension of the latent noise vector.
        device: Device to use ('cpu' or 'cuda').

    Returns:
        The generator loss.
    """

    from src.utils import get_noise

    # Generate fake images
    noise: torch.Tensor = get_noise(num_images, z_dim, device)
    fake_images: torch.Tensor = gen(noise)

    # Get discriminator output
    fake_output: torch.Tensor = disc(fake_images)

    # Generator wants discriminator to output 1 (real) for fake images
    real_target: torch.Tensor = torch.ones(num_images, 1, device=device)
    gen_loss: torch.Tensor = criterion(fake_output, real_target)

    return gen_loss
