import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

try:
    from .vq_vae import VQVAE
except ImportError:
    from vq_vae import VQVAE


def load_checkpoint(checkpoint_path, device="cpu"):
    """
    Load VQ-VAE model from checkpoint file.

    Args:
        checkpoint_path (str): Path to the .pt checkpoint file
        device (str): Device to load the model on ('cpu' or 'cuda')

    Returns:
        tuple: (model, checkpoint_info) where checkpoint_info contains
               training metadata like iteration, losses, etc.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Hyperparameters
    model_config = {
        "num_hiddens": 128,
        "num_residual_layers": 2,
        "num_residual_hiddens": 32,
        "num_embeddings": 512,
        "embedding_dim": 64,
        "commitment_cost": 0.25,
        "decay": 0.99,
    }

    # Create model
    model = VQVAE(**model_config)

    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Extract checkpoint info
    checkpoint_info = {
        "iteration": checkpoint.get("iteration", "unknown"),
        "recon_error": checkpoint.get("recon_error", []),
        "perplexity": checkpoint.get("perplexity", []),
    }

    return model, checkpoint_info


def test_reconstruction(model, data_loader, device="cpu", num_samples=8):
    """
    Test model reconstruction quality on a batch of data.

    Args:
        model: Trained VQ-VAE model
        data_loader: DataLoader with test data
        device: Device to run inference on
        num_samples: Number of samples to reconstruct

    Returns:
        tuple: (original_images, reconstructed_images, losses)
    """
    model.eval()

    with torch.no_grad():
        # Get a batch of data
        data_iter = iter(data_loader)
        original_images, _ = next(data_iter)
        original_images = original_images[:num_samples].to(device)

        # Forward pass
        vq_loss, reconstructed_images, perplexity = model(original_images)
        recon_loss = F.mse_loss(reconstructed_images, original_images)

        losses = {
            "vq_loss": vq_loss.item(),
            "recon_loss": recon_loss.item(),
            "perplexity": perplexity.item(),
        }

        return original_images.cpu(), reconstructed_images.cpu(), losses


def sample_from_codebook(model, device="cpu", num_samples=16, image_shape=(3, 32, 32)):
    """
    Generate samples by randomly sampling from the learned codebook.

    Args:
        model: Trained VQ-VAE model
        device: Device to run inference on
        num_samples: Number of samples to generate
        image_shape: Expected output image shape (C, H, W)

    Returns:
        torch.Tensor: Generated samples
    """
    model.eval()

    with torch.no_grad():
        # Get codebook embeddings
        codebook = (
            model._vq_vae._embedding.weight
        )  # Shape: (num_embeddings, embedding_dim)
        K, D = codebook.shape

        # Calculate latent spatial dimensions (assuming 4x downsampling)
        latent_h, latent_w = image_shape[1] // 4, image_shape[2] // 4

        # Sample random indices from codebook
        indices = torch.randint(0, K, (num_samples, latent_h, latent_w), device=device)

        # Map indices to embeddings
        z_q = codebook[indices]  # (num_samples, latent_h, latent_w, embedding_dim)
        z_q = z_q.permute(
            0, 3, 1, 2
        ).contiguous()  # (num_samples, embedding_dim, latent_h, latent_w)

        # Decode samples
        samples = model._decoder(z_q)

        return samples.cpu()


def visualize_results(original, reconstructed, samples=None, save_path=None):
    """
    Visualize original images, reconstructions, and optionally generated samples.

    Args:
        original: Original images tensor
        reconstructed: Reconstructed images tensor
        samples: Optional generated samples tensor
        save_path: Optional path to save the visualization
    """
    fig_width = 15 if samples is not None else 10
    fig, axes = plt.subplots(1, 3 if samples is not None else 2, figsize=(fig_width, 5))

    # Normalize images to [0, 1] for display
    def normalize_for_display(img_tensor):
        img = img_tensor.clone()
        img = torch.clamp(img, -1, 1)  # Assuming images are in [-1, 1]
        img = (img + 1) / 2  # Convert to [0, 1]
        return img

    # Original images
    original_grid = make_grid(normalize_for_display(original), nrow=4, padding=2)
    axes[0].imshow(original_grid.permute(1, 2, 0))
    axes[0].set_title("Original Images")
    axes[0].axis("off")

    # Reconstructed images
    recon_grid = make_grid(normalize_for_display(reconstructed), nrow=4, padding=2)
    axes[1].imshow(recon_grid.permute(1, 2, 0))
    axes[1].set_title("Reconstructions")
    axes[1].axis("off")

    # Generated samples (if provided)
    if samples is not None:
        samples_grid = make_grid(normalize_for_display(samples), nrow=4, padding=2)
        axes[2].imshow(samples_grid.permute(1, 2, 0))
        axes[2].set_title("Generated Samples")
        axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to {save_path}")

    plt.show()


def create_test_dataloader(dataset_name="CIFAR10", batch_size=32, data_path="./data"):
    """
    Create a test data loader for evaluation.

    Args:
        dataset_name: Name of dataset ('CIFAR10' or 'CIFAR100')
        batch_size: Batch size for data loader
        data_path: Path to store/load dataset

    Returns:
        DataLoader: Test data loader
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))]
    )

    if dataset_name == "CIFAR10":
        test_dataset = datasets.CIFAR10(
            root=data_path, train=False, download=True, transform=transform
        )
    elif dataset_name == "CIFAR100":
        test_dataset = datasets.CIFAR100(
            root=data_path, train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )

    return test_loader


def evaluate_checkpoint(
    checkpoint_path, dataset_name="CIFAR10", device=None, save_viz=True
):
    """
    Complete evaluation pipeline for a VQ-VAE checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file
        dataset_name: Dataset to evaluate on
        device: Device to use (auto-detected if None)
        save_viz: Whether to save visualization

    Returns:
        dict: Evaluation results including losses and model info
    """
    # Auto-detect device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Using device: {device}")

    # Load model
    model, checkpoint_info = load_checkpoint(checkpoint_path, device)
    print(f"Model loaded from iteration: {checkpoint_info['iteration']}")

    # Create test data loader
    test_loader = create_test_dataloader(dataset_name)

    # Test reconstruction
    print("Testing reconstruction...")
    original, reconstructed, losses = test_reconstruction(model, test_loader, device)

    print(f"Reconstruction Loss: {losses['recon_loss']:.4f}")
    print(f"VQ Loss: {losses['vq_loss']:.4f}")
    print(f"Perplexity: {losses['perplexity']:.2f}")

    # Generate samples
    print("Generating samples from codebook...")
    samples = sample_from_codebook(model, device)

    # Visualize results
    save_path = (
        f"evaluation_results_{checkpoint_info['iteration']}.png" if save_viz else None
    )
    visualize_results(original, reconstructed, samples, save_path)

    return {"losses": losses, "checkpoint_info": checkpoint_info, "model": model}