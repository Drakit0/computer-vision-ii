"""
Script for generating the train loop, and functions of train and test.

This script implements the GAN training pipeline including:
- Data loading and preprocessing
- Training loop for both discriminator and generator
- Loss tracking and visualization
- Model checkpointing
"""

import os
import sys

# Add project root to path to enable src imports when running as script
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, random_split  # noqa: E402
from torchvision import transforms  # noqa: E402
from torchvision.datasets import MNIST  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.model import Discriminator, Generator  # noqa: E402
from src.utils import (  # noqa: E402
    get_noise,
    save_gan_losses,
    save_generated_samples,
    set_seed,
)

# Training hyperparameters
N_EPOCHS: int = 200
Z_DIM: int = 64
DISPLAY_STEP: int = 500
BATCH_SIZE: int = 128
LR: float = 0.0002
BETA_1: float = 0.5
BETA_2: float = 0.999
TRAIN_SPLIT: float = 0.9
PATH_WEIGHTS_GEN: str = "weights/generator.pt"
PATH_WEIGHTS_DISC: str = "weights/discriminator.pt"


class Trainer:
    """
    Class of the trainer for the GAN model.
    """

    def __init__(
        self,
        z_dim: int = Z_DIM,
        batch_size: int = BATCH_SIZE,
        lr: float = LR,
        beta_1: float = BETA_1,
        beta_2: float = BETA_2,
        seed: int = 42,
    ) -> None:
        """
        Constructor of the class.

        Args:
            z_dim: Dimension of the latent noise vector.
            batch_size: Batch size for training.
            lr: Learning rate.
            beta_1: Beta1 for Adam optimizer.
            beta_2: Beta2 for Adam optimizer.
            seed: Random seed for reproducibility.
        """

        # Set random seeds for reproducibility
        set_seed(seed)

        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.z_dim = z_dim
        self.batch_size = batch_size

        # Get data loaders
        self.train_loader, self.test_loader = self._get_loaders()

        # Initialize models
        self.generator = Generator(latent_dim=z_dim).to(self.device)
        self.discriminator = Discriminator().to(self.device)

        # Initialize optimizers
        self.gen_opt = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(beta_1, beta_2)
        )
        self.disc_opt = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(beta_1, beta_2)
        )

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()

        # Loss tracking
        self.losses: dict[str, list[float]] = {
            "d_real": [],
            "d_fake": [],
            "d_total": [],
            "g": [],
        }

        # Fixed noise for visualization
        self.fixed_noise: torch.Tensor = get_noise(64, z_dim, str(self.device))

        # Generated samples per epoch for visualization
        self.generated_samples: list[torch.Tensor] = []

    def _get_loaders(self) -> tuple[DataLoader, DataLoader]:
        """
        Obtains the train and test data loaders.

        Returns:
            Train and test data loader.
        """

        # Download and load MNIST dataset
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        full_dataset = MNIST(
            root="data",
            train=True,
            download=True,
            transform=transform,
        )

        # Split into train and validation
        train_size: int = int(TRAIN_SPLIT * len(full_dataset))
        val_size: int = len(full_dataset) - train_size

        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        train_loader: DataLoader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

        test_loader: DataLoader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
        )

        return train_loader, test_loader

    def _train_discriminator(self, real_images: torch.Tensor) -> torch.Tensor:
        """
        Performs one training step for the discriminator.

        Args:
            real_images: Batch of real images. Dimensions: [batch, 1, 28, 28].

        Returns:
            Discriminator loss.
        """

        self.disc_opt.zero_grad()

        batch_size: int = real_images.shape[0]

        # Loss on real images
        real_output: torch.Tensor = self.discriminator(real_images)
        real_target: torch.Tensor = torch.ones(batch_size, 1, device=self.device)
        real_loss: torch.Tensor = self.criterion(real_output, real_target)

        # Generate fake images
        noise: torch.Tensor = get_noise(batch_size, self.z_dim, str(self.device))
        fake_images: torch.Tensor = self.generator(noise).detach()

        # Loss on fake images
        fake_output: torch.Tensor = self.discriminator(fake_images)
        fake_target: torch.Tensor = torch.zeros(batch_size, 1, device=self.device)
        fake_loss: torch.Tensor = self.criterion(fake_output, fake_target)

        # Total discriminator loss
        disc_loss: torch.Tensor = (real_loss + fake_loss) / 2

        # Backpropagate
        disc_loss.backward()
        self.disc_opt.step()

        # Track losses
        self.losses["d_real"].append(float(real_loss.detach()))
        self.losses["d_fake"].append(float(fake_loss.detach()))
        self.losses["d_total"].append(float(disc_loss.detach()))

        return disc_loss

    def _train_generator(self, batch_size: int) -> torch.Tensor:
        """
        Performs one training step for the generator.

        Args:
            batch_size: Number of images to generate.

        Returns:
            Generator loss.
        """

        self.gen_opt.zero_grad()

        # Generate fake images
        noise: torch.Tensor = get_noise(batch_size, self.z_dim, str(self.device))
        fake_images: torch.Tensor = self.generator(noise)

        # Get discriminator output
        fake_output: torch.Tensor = self.discriminator(fake_images)

        # Generator wants discriminator to output 1 (real) for fake images
        real_target: torch.Tensor = torch.ones(batch_size, 1, device=self.device)
        gen_loss: torch.Tensor = self.criterion(fake_output, real_target)

        # Backpropagate
        gen_loss.backward()
        self.gen_opt.step()

        # Track loss
        self.losses["g"].append(float(gen_loss.detach()))

        return gen_loss

    def _generate_samples(self) -> torch.Tensor:
        """
        Generates samples using the fixed noise for visualization.

        Returns:
            Generated images. Dimensions: [64, 1, 28, 28].
        """

        self.generator.eval()
        with torch.no_grad():
            samples: torch.Tensor = self.generator(self.fixed_noise)
        self.generator.train()

        return samples

    def _evaluate(self) -> tuple[float, float]:
        """
        Evaluates the model on the test set.

        Returns:
            Average discriminator and generator loss on test set.
        """

        self.discriminator.eval()
        self.generator.eval()

        total_disc_loss: float = 0.0
        total_gen_loss: float = 0.0
        num_batches: int = 0

        with torch.no_grad():
            for real_images, _ in self.test_loader:
                real_images = real_images.to(self.device)
                batch_size = real_images.shape[0]

                # Discriminator loss
                real_output: torch.Tensor = self.discriminator(real_images)
                real_target: torch.Tensor = torch.ones(
                    batch_size, 1, device=self.device
                )
                real_loss: torch.Tensor = self.criterion(real_output, real_target)

                noise: torch.Tensor = get_noise(
                    batch_size, self.z_dim, str(self.device)
                )
                fake_images: torch.Tensor = self.generator(noise)

                fake_output: torch.Tensor = self.discriminator(fake_images)
                fake_target: torch.Tensor = torch.zeros(
                    batch_size, 1, device=self.device
                )
                fake_loss: torch.Tensor = self.criterion(fake_output, fake_target)

                disc_loss: torch.Tensor = (real_loss + fake_loss) / 2
                total_disc_loss += float(disc_loss)

                # Generator loss
                gen_target: torch.Tensor = torch.ones(batch_size, 1, device=self.device)
                gen_loss: torch.Tensor = self.criterion(fake_output, gen_target)
                total_gen_loss += float(gen_loss)

                num_batches += 1

        self.discriminator.train()
        self.generator.train()

        return total_disc_loss / num_batches, total_gen_loss / num_batches

    def fit(
        self,
        n_epochs: int = N_EPOCHS,
        display_step: int = DISPLAY_STEP,
        save_every: int = 10,
        path_weights_gen: str = PATH_WEIGHTS_GEN,
        path_weights_disc: str = PATH_WEIGHTS_DISC,
        path_images: str = "images",
    ) -> tuple[Discriminator, Generator]:
        """
        Training of the GAN. Saves the parameters of both models.

        Args:
            n_epochs: Number of epochs to train.
            display_step: How often to display/visualize the images.
            save_every: To save the model every 'save_every' epochs.
            path_weights_gen: Path where the generator weights are saved.
            path_weights_disc: Path where the discriminator weights are saved.
            path_images: Path where the images are saved.

        Returns:
            Trained discriminator and generator.
        """

        # Create directories
        os.makedirs(os.path.dirname(path_weights_gen), exist_ok=True)
        os.makedirs(path_images, exist_ok=True)

        step: int = 0

        for epoch in range(n_epochs):

            # Save generated samples at the start of each epoch
            samples = self._generate_samples()
            self.generated_samples.append(samples.cpu())

            for real_images, _ in tqdm(
                self.train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}"
            ):
                real_images = real_images.to(self.device)
                batch_size: int = real_images.shape[0]

                # Update discriminator
                self._train_discriminator(real_images)

                # Update generator
                self._train_generator(batch_size)

                # Display progress
                if step % display_step == 0 and step > 0:
                    avg_d_loss: float = sum(self.losses["d_total"][-display_step:]) / (
                        display_step
                    )
                    avg_g_loss: float = sum(self.losses["g"][-display_step:]) / (
                        display_step
                    )
                    print(
                        f"\nStep {step}: D Loss: {avg_d_loss:.4f}, "
                        f"G Loss: {avg_g_loss:.4f}"
                    )

                step += 1

            # Evaluate on test set
            test_d_loss, test_g_loss = self._evaluate()
            print(
                f"Epoch {epoch + 1}: Test D Loss: {test_d_loss:.4f}, "
                f"Test G Loss: {test_g_loss:.4f}"
            )

            # Save model every few epochs
            if (epoch + 1) % save_every == 0:
                torch.save(self.generator.state_dict(), path_weights_gen)
                torch.save(self.discriminator.state_dict(), path_weights_disc)
                print(f"Saved models at epoch {epoch + 1}")

        # Save final models
        torch.save(self.generator.state_dict(), path_weights_gen)
        torch.save(self.discriminator.state_dict(), path_weights_disc)

        # Save loss plots
        save_gan_losses(
            self.losses["d_real"],
            self.losses["d_fake"],
            self.losses["d_total"],
            self.losses["g"],
            f"{path_images}/gan_losses.png",
        )

        # Save generated samples evolution
        epochs_to_show: list[int] = [
            0,
            n_epochs // 4,
            n_epochs // 2,
            3 * n_epochs // 4,
            n_epochs - 1,
        ]
        epochs_to_show = [e for e in epochs_to_show if e < len(self.generated_samples)]
        save_generated_samples(
            epochs_to_show,
            self.generated_samples,
            samples_per_epoch=8,
            path=f"{path_images}/generated_samples_per_epoch.png",
        )

        return self.discriminator, self.generator


def train(
    n_epochs: int = 5,
    z_dim: int = Z_DIM,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    seed: int = 42,
) -> tuple[Discriminator, Generator]:
    """
    Trains a GAN and returns the trained discriminator and generator.

    This is a convenience function for training with default or custom parameters.

    Args:
        n_epochs: Number of epochs to train.
        z_dim: Dimension of the latent noise vector.
        batch_size: Batch size for training.
        lr: Learning rate.
        seed: Random seed for reproducibility.

    Returns:
        Trained discriminator and generator.
    """

    trainer = Trainer(
        z_dim=z_dim,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
    )

    discriminator, generator = trainer.fit(n_epochs=n_epochs)

    return discriminator, generator


if __name__ == "__main__":
    # Set your parameters
    set_seed(42)

    trainer = Trainer(
        z_dim=Z_DIM,
        batch_size=BATCH_SIZE,
        lr=LR,
    )

    discriminator, generator = trainer.fit(
        n_epochs=N_EPOCHS,
        display_step=DISPLAY_STEP,
    )

    print("Training completed!")
