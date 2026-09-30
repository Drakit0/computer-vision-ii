import torch
import torch.nn as nn


def update_ema_weights(
    ema_cluster_size: torch.Tensor,
    ema_w: nn.Parameter,
    embedding_weight: torch.Tensor,
    encodings: torch.Tensor,
    flat_input: torch.Tensor,
    decay: float,
    epsilon: float,
    num_embeddings: int,
) -> None:
    """Update EMA weights for vector quantizer.

    This function implements the exponential moving average updates for the
    VQ-VAE codebook. It updates both the cluster sizes and the embedding weights
    using EMA, which often leads to more stable training than gradient-based updates.

    Args:
        ema_cluster_size: EMA cluster size buffer [num_embeddings].
        ema_w: EMA weight parameter [num_embeddings, embedding_dim].
        embedding_weight: Embedding weight parameter [num_embeddings, embedding_dim].
        encodings: One-hot encodings [batch_size, num_embeddings].
        flat_input: Flattened input tensor [batch_size, embedding_dim].
        decay: EMA decay rate (typically 0.99).
        epsilon: Smoothing parameter for Laplace smoothing.
        num_embeddings: Number of embeddings in codebook.
    """
    # Update cluster sizes using EMA
    # encodings.sum(0) gives the count of assignments to each embedding
    ema_cluster_size.data.mul_(decay).add_(encodings.sum(0), alpha=1 - decay)

    # Update embedding weights using EMA
    # encodings.t() @ flat_input gives the sum of all inputs assigned to each embedding
    dw = encodings.t() @ flat_input
    ema_w.data.mul_(decay).add_(dw, alpha=1 - decay)

    # Apply Laplace smoothing to prevent division by zero for unused embeddings
    n = ema_cluster_size.sum()
    cluster_size_smoothed = (
        (ema_cluster_size + epsilon) / (n + num_embeddings * epsilon) * n
    )

    # Normalize embedding weights by cluster sizes
    embedding_weight.data.copy_(ema_w / cluster_size_smoothed.unsqueeze(1))