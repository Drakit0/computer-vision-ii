from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    from .config import VQVAEConfig
    from .training import train_one_epoch
    from .vq_vae import VQVAE
except ImportError:
    from config import VQVAEConfig
    from training import train_one_epoch
    from vq_vae import VQVAE


def create_dataloader(
    dataset_name: str = "CIFAR10", batch_size: int = 256, data_path: str = "./data"
) -> Tuple[DataLoader, DataLoader, float]:
    """Create training and validation dataloaders.

    Args:
        dataset_name: Name of dataset to load (currently supports "CIFAR10").
        batch_size: Batch size for training.
        data_path: Path to store/load data.

    Returns:
        Tuple of (train_loader, val_loader, data_variance).
    """
    # Define transforms
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
        ]
    )

    # Load dataset
    if dataset_name == "CIFAR10":
        train_dataset = datasets.CIFAR10(
            root=data_path, train=True, download=True, transform=transform
        )
        val_dataset = datasets.CIFAR10(
            root=data_path, train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Compute data variance for normalization
    # For normalized CIFAR-10, variance is approximately 1.0
    # But we compute it more accurately from the data
    data_variance = np.var(train_dataset.data / 255.0)

    return train_loader, val_loader, float(data_variance)


def fit(config: VQVAEConfig) -> Tuple[VQVAE, List[float], List[float]]:
    """Train VQ-VAE model using configuration.

    Args:
        config: VQVAEConfig containing all training parameters.

    Returns:
        Tuple of (trained_model, reconstruction_errors, perplexities).
    """
    # Setup device
    device = config.device

    # Create dataloaders
    train_loader, val_loader, data_variance = create_dataloader(
        dataset_name=config.dataset_name,
        batch_size=config.batch_size,
        data_path=config.data_path,
    )

    # Create model
    model = VQVAE(
        num_hiddens=config.num_hiddens,
        num_residual_layers=config.num_residual_layers,
        num_residual_hiddens=config.num_residual_hiddens,
        num_embeddings=config.num_embeddings,
        embedding_dim=config.embedding_dim,
        commitment_cost=config.commitment_cost,
        decay=config.decay,
    ).to(device)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training history
    recon_errors: List[float] = []
    perplexities: List[float] = []

    # Calculate number of epochs based on iterations
    iterations_per_epoch = len(train_loader)
    num_epochs = max(1, config.num_training_updates // iterations_per_epoch)

    # Create save directory
    save_path = Path(config.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Training loop
    iteration = 0
    for epoch in range(num_epochs):
        # Train one epoch
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            data_variance=data_variance,
            use_amp=config.use_amp,
        )

        # Track metrics
        recon_errors.append(train_metrics["recon_loss"])
        perplexities.append(train_metrics["perplexity"])

        # Update iteration count
        iteration += iterations_per_epoch

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Loss: {train_metrics['loss']:.4f} | "
                f"Recon: {train_metrics['recon_loss']:.4f} | "
                f"Perplexity: {train_metrics['perplexity']:.2f}"
            )

        # Save checkpoint periodically
        if iteration >= config.num_training_updates:
            break

    # Save final model
    checkpoint = {
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "recon_error": recon_errors,
        "perplexity": perplexities,
    }
    torch.save(checkpoint, save_path / f"model_{iteration}.pt")

    return model, recon_errors, perplexities


if __name__ == "__main__":
    config = VQVAEConfig()
    fit(config)