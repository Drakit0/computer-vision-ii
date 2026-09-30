from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class VQVAEConfig:
    """Configuration for VQ-VAE model and training."""

    # Model architecture
    num_hiddens: int = 128
    num_residual_layers: int = 2
    num_residual_hiddens: int = 32
    embedding_dim: int = 64
    num_embeddings: int = 512
    commitment_cost: float = 0.25
    decay: float = 0.99

    # Training parameters
    num_training_updates: int = 15000
    batch_size: int = 256
    learning_rate: float = 1e-3
    use_amp: bool = True

    # Data and device
    dataset_name: str = "CIFAR10"
    data_path: str = "./data"
    save_path: str = "./model"
    device: Optional[torch.device] = None

    def __post_init__(self):
        """Set default device if not specified."""
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")