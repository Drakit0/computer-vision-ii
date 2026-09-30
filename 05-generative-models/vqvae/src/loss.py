import torch
import torch.nn.functional as F


def vq_loss(
    quantized: torch.Tensor, inputs: torch.Tensor, commitment_cost: float
) -> torch.Tensor:
    """Vector quantization loss with commitment term.

    The VQ loss consists of two components:
    1. Codebook loss: Updates the embeddings to be closer to encoder outputs.
    2. Commitment loss: Encourages encoder outputs to commit to embeddings.

    The total loss is: codebook_loss + commitment_cost * commitment_loss

    Args:
        quantized: Quantized latent vectors (output of quantization).
        inputs: Original encoder outputs (input to quantization).
        commitment_cost: Weight for commitment loss term.

    Returns:
        Combined VQ loss tensor (scalar).
    """
    # Codebook loss: move embeddings towards encoder outputs
    # Stop gradient on inputs so only embeddings are updated
    codebook_loss = F.mse_loss(quantized, inputs.detach())

    # Commitment loss: encourage encoder to commit to embeddings
    # Stop gradient on quantized so only encoder is updated
    commitment_loss = F.mse_loss(quantized.detach(), inputs)

    return codebook_loss + commitment_cost * commitment_loss


def vq_ema_loss(
    quantized: torch.Tensor, inputs: torch.Tensor, commitment_cost: float
) -> torch.Tensor:
    """Vector quantization loss for EMA version.

    For EMA quantizers, only the commitment loss is needed since
    the codebook is updated via exponential moving averages.

    Args:
        quantized: Quantized latent vectors.
        inputs: Original encoder outputs.
        commitment_cost: Weight for commitment loss.

    Returns:
        EMA VQ loss tensor (scalar).
    """
    # Only commitment loss for EMA version
    commitment_loss = F.mse_loss(quantized.detach(), inputs)

    return commitment_cost * commitment_loss


def reconstruction_loss(
    recon: torch.Tensor, target: torch.Tensor, data_variance: float = 1.0
) -> torch.Tensor:
    """Reconstruction loss normalized by data variance.

    This loss measures how well the decoder reconstructs the original input.
    Normalizing by data variance helps with training stability.

    Args:
        recon: Reconstructed images from decoder.
        target: Original input images.
        data_variance: Variance of the training data for normalization.

    Returns:
        Normalized reconstruction loss (scalar).
    """
    return F.mse_loss(recon, target) / data_variance