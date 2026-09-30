import matplotlib.pyplot as plt
import torch
from scipy.signal import savgol_filter
from torchvision.utils import make_grid


def plot_training_curves(recon_errors, perplexities, save_path=None):
    """Plot training curves for reconstruction error and perplexity."""
    recon_smooth = savgol_filter(recon_errors, 201, 7)
    perp_smooth = savgol_filter(perplexities, 201, 7)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    ax1.plot(recon_smooth)
    ax1.set_yscale("log")
    ax1.set_title("Smoothed NMSE")
    ax1.set_xlabel("iteration")

    ax2.plot(perp_smooth)
    ax2.set_title("Smoothed Average codebook usage (perplexity)")
    ax2.set_xlabel("iteration")

    if save_path:
        plt.savefig(save_path)
    plt.show()


def show_reconstructions(model, data_loader, device, save_path=None):
    """Show original vs reconstructed images."""
    model.eval()

    with torch.no_grad():
        (data, _) = next(iter(data_loader))
        data = data.to(device)

        vq_loss, reconstructions, perplexity = model(data)

        # Normalize for display
        data_display = (data + 1) / 2
        recon_display = torch.clamp((reconstructions + 1) / 2, 0, 1)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))

        # Original
        grid_orig = make_grid(data_display.cpu(), nrow=8, padding=2)
        ax1.imshow(grid_orig.permute(1, 2, 0))
        ax1.set_title("Original Images")
        ax1.axis("off")

        # Reconstructions
        grid_recon = make_grid(recon_display.cpu(), nrow=8, padding=2)
        ax2.imshow(grid_recon.permute(1, 2, 0))
        ax2.set_title("Reconstructions")
        ax2.axis("off")

        if save_path:
            plt.savefig(save_path)
        plt.show()


def visualize_codebook(model, save_path=None):
    """Visualize the learned codebook using UMAP."""
    try:
        import umap

        embeddings = model._vq_vae._embedding.weight.data.cpu().numpy()
        reducer = umap.UMAP(n_neighbors=3, min_dist=0.1, metric="cosine")
        embedding_2d = reducer.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.6)
        plt.title("Codebook Embeddings (UMAP)")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")

        if save_path:
            plt.savefig(save_path)
        plt.show()

    except ImportError:
        print("UMAP not available. Install with: pip install umap-learn")


def generate_samples(model, device, num_samples=16, save_path=None):
    """Generate samples by randomly sampling from codebook."""
    model.eval()

    with torch.no_grad():
        codebook = model._vq_vae._embedding.weight
        K, D = codebook.shape

        # Sample random indices
        indices = torch.randint(0, K, (num_samples, 8, 8), device=device)
        z_q = codebook[indices].permute(0, 3, 1, 2)

        # Decode
        samples = model._decoder(z_q)
        samples_display = torch.clamp((samples + 1) / 2, 0, 1)

        # Show
        grid = make_grid(samples_display.cpu(), nrow=4, padding=2)
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0))
        plt.title("Generated Samples")
        plt.axis("off")

        if save_path:
            plt.savefig(save_path)
        plt.show()


def analyze_model(checkpoint_path, device=None):
    """Complete analysis of a trained model."""
    try:
        from .checkpoint_utils import create_test_dataloader, load_checkpoint
    except ImportError:
        from checkpoint_utils import create_test_dataloader, load_checkpoint

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model, info = load_checkpoint(checkpoint_path, device)

    # Create data loader
    test_loader = create_test_dataloader()

    print(f"Model from iteration: {info['iteration']}")

    # Plot training curves if available
    if info["recon_error"] and info["perplexity"]:
        plot_training_curves(
            info["recon_error"], info["perplexity"], "training_curves.png"
        )

    # Show reconstructions
    show_reconstructions(model, test_loader, device, "reconstructions.png")

    # Generate samples
    generate_samples(model, device, save_path="samples.png")

    # Visualize codebook
    visualize_codebook(model, "codebook.png")

    print("Analysis complete! Check generated images.")