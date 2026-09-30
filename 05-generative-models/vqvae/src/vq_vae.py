from typing import Tuple

import torch
from torch import nn

try:
    from .ema_update import update_ema_weights
    from .loss import vq_ema_loss, vq_loss
    from .networks import Decoder, Encoder
except ImportError:
    from ema_update import update_ema_weights
    from loss import vq_ema_loss, vq_loss
    from networks import Decoder, Encoder


class VectorQuantizer(nn.Module):
    """Vector Quantization layer with commitment loss.

    This layer performs vector quantization by finding the closest embedding
    for each input vector and applying straight-through estimation.
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, commitment_cost: float
    ) -> None:
        """Initialize vector quantizer.

        Args:
            num_embeddings: Size of the codebook (number of discrete codes).
            embedding_dim: Dimension of each embedding vector.
            commitment_cost: Weight for the commitment loss term.
        """
        super().__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        # Initialize embedding layer (codebook)
        self._embedding = nn.Embedding(num_embeddings, embedding_dim)
        self._embedding.weight.data.uniform_(
            -1.0 / num_embeddings, 1.0 / num_embeddings
        )

        self._commitment_cost = commitment_cost

    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through vector quantizer.

        Args:
            inputs: Input tensor [B, D, H, W] where D is embedding_dim.

        Returns:
            Tuple of (loss, quantized, perplexity, encodings).
        """
        # Reshape inputs from [B, D, H, W] to [B*H*W, D]
        inputs_shape = inputs.shape
        flat_input = inputs.permute(0, 2, 3, 1).contiguous()
        flat_input = flat_input.view(-1, self._embedding_dim)

        # Compute distances: ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z*e
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Find closest embeddings (argmin of distances)
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # Create one-hot encodings
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize using embeddings
        quantized = torch.matmul(encodings, self._embedding.weight)
        quantized = quantized.view(
            inputs_shape[0], inputs_shape[2], inputs_shape[3], self._embedding_dim
        )

        # Compute VQ loss
        loss = vq_loss(quantized, flat_input.view_as(quantized), self._commitment_cost)

        # Straight-through estimator: copy gradients from quantized to inputs
        quantized = inputs + (quantized.permute(0, 3, 1, 2) - inputs).detach()

        # Compute perplexity (measure of codebook usage)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized, perplexity, encodings


class VectorQuantizerEMA(nn.Module):
    """Vector Quantization layer with Exponential Moving Average updates.

    This version updates the codebook using EMA instead of gradient descent,
    which often leads to more stable training.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float,
        decay: float,
        epsilon: float = 1e-5,
    ) -> None:
        """Initialize EMA vector quantizer.

        Args:
            num_embeddings: Size of the codebook.
            embedding_dim: Dimension of each embedding vector.
            commitment_cost: Weight for the commitment loss term.
            decay: EMA decay rate (typically 0.99).
            epsilon: Small constant for numerical stability.
        """
        super().__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        # Initialize embedding layer
        self._embedding = nn.Embedding(num_embeddings, embedding_dim)
        self._embedding.weight.data.normal_()

        self._commitment_cost = commitment_cost

        # Register EMA buffers
        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through EMA vector quantizer.

        Args:
            inputs: Input tensor [B, D, H, W].

        Returns:
            Tuple of (loss, quantized, perplexity, encodings).
        """
        # Reshape inputs from [B, D, H, W] to [B*H*W, D]
        inputs_shape = inputs.shape
        flat_input = inputs.permute(0, 2, 3, 1).contiguous()
        flat_input = flat_input.view(-1, self._embedding_dim)

        # Compute distances: ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z*e
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Find closest embeddings
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # Create one-hot encodings
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize using embeddings
        quantized = torch.matmul(encodings, self._embedding.weight)
        quantized = quantized.view(
            inputs_shape[0], inputs_shape[2], inputs_shape[3], self._embedding_dim
        )

        # Update EMA weights during training
        if self.training:
            update_ema_weights(
                self._ema_cluster_size,
                self._ema_w,
                self._embedding.weight,
                encodings,
                flat_input,
                self._decay,
                self._epsilon,
                self._num_embeddings,
            )

        # Compute EMA loss (commitment loss only)
        loss = vq_ema_loss(
            quantized, flat_input.view_as(quantized), self._commitment_cost
        )

        # Straight-through estimator
        quantized = inputs + (quantized.permute(0, 3, 1, 2) - inputs).detach()

        # Compute perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized, perplexity, encodings


class VQVAE(nn.Module):
    """Complete VQ-VAE model combining encoder, quantizer, and decoder."""

    def __init__(
        self,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float,
        decay: float = 0,
    ) -> None:
        """Initialize VQ-VAE model.

        Args:
            num_hiddens: Number of hidden channels in encoder/decoder.
            num_residual_layers: Number of residual layers.
            num_residual_hiddens: Hidden channels in residual blocks.
            num_embeddings: Size of codebook.
            embedding_dim: Dimension of embeddings.
            commitment_cost: Weight for commitment loss.
            decay: EMA decay rate (0 for standard VQ, >0 for EMA).
        """
        super().__init__()

        # Encoder
        self._encoder = Encoder(
            in_channels=3,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )

        # Pre-quantization convolution: project to embedding dimension
        self._pre_vq_conv = nn.Conv2d(
            in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1
        )

        # Vector quantizer (use EMA if decay > 0)
        if decay > 0:
            self._vq_vae = VectorQuantizerEMA(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                commitment_cost=commitment_cost,
                decay=decay,
            )
        else:
            self._vq_vae = VectorQuantizer(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                commitment_cost=commitment_cost,
            )

        # Decoder
        self._decoder = Decoder(
            in_channels=embedding_dim,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VQ-VAE.

        Args:
            x: Input images [B, C, H, W].

        Returns:
            Tuple of (vq_loss, reconstruction, perplexity).
        """
        # Encode
        z = self._encoder(x)

        # Project to embedding dimension
        z = self._pre_vq_conv(z)

        # Quantize
        loss, quantized, perplexity, _ = self._vq_vae(z)

        # Decode
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity