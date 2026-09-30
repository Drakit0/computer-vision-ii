import torch
from torch import nn, optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

try:
    from .loss import reconstruction_loss
except ImportError:
    from loss import reconstruction_loss


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device | None,
    data_variance: float,
    use_amp: bool = True,
) -> dict[str, float]:
    """Train model for one epoch.

    Args:
        model: VQ-VAE model to train.
        dataloader: Training data loader.
        optimizer: Model optimizer (typically Adam).
        device: Training device (CPU or CUDA).
        data_variance: Data variance for loss normalization.
        use_amp: Whether to use Automatic Mixed Precision.

    Returns:
        Dictionary with training metrics.
    """
    model.train()

    # Initialize metrics
    total_loss = 0.0
    total_recon_loss = 0.0
    total_perplexity = 0.0
    num_batches = 0

    # Initialize GradScaler for AMP if using CUDA
    use_amp_cuda = use_amp and device is not None and device.type == "cuda"
    scaler = GradScaler() if use_amp_cuda else None

    for data, _ in dataloader:
        data = data.to(device)

        optimizer.zero_grad()

        if use_amp_cuda and scaler is not None:
            with autocast(device_type="cuda"):
                vq_loss, recon, perplexity = model(data)
                recon_loss = reconstruction_loss(recon, data, data_variance)
                loss = recon_loss + vq_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            vq_loss, recon, perplexity = model(data)
            recon_loss = reconstruction_loss(recon, data, data_variance)
            loss = recon_loss + vq_loss

            loss.backward()
            optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_perplexity += perplexity.item()
        num_batches += 1

    # Average metrics
    return {
        "loss": total_loss / num_batches,
        "recon_loss": total_recon_loss / num_batches,
        "perplexity": total_perplexity / num_batches,
    }


def eval_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device | None,
    data_variance: float,
) -> dict[str, float]:
    """Evaluate model for one epoch.

    Args:
        model: VQ-VAE model to evaluate.
        dataloader: Validation/test data loader.
        device: Evaluation device.
        data_variance: Data variance for loss normalization.

    Returns:
        Dictionary with evaluation metrics.
    """
    model.eval()

    # Initialize metrics
    total_loss = 0.0
    total_recon_loss = 0.0
    total_perplexity = 0.0
    num_batches = 0

    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)

            vq_loss, recon, perplexity = model(data)
            recon_loss = reconstruction_loss(recon, data, data_variance)
            loss = recon_loss + vq_loss

            # Accumulate metrics
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_perplexity += perplexity.item()
            num_batches += 1

    # Average metrics
    return {
        "loss": total_loss / num_batches,
        "recon_loss": total_recon_loss / num_batches,
        "perplexity": total_perplexity / num_batches,
    }