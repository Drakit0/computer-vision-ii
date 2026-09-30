"""
Script to train the model.
"""

import os
import sys

# Add project root to path to enable src imports when running as script
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.constants import BATCH_SIZE, EPOCHS, LR, PATH_MODEL  # noqa: E402
from src.data.dataset import RaccoonDataset  # noqa: E402
from src.model.loss import YOLOLoss  # noqa: E402
from src.model.model import YOLO  # noqa: E402
from src.utils import mean_average_precision  # noqa: E402


class Trainer:
    """
    Class of the trainer of the model.
    """

    def __init__(self, seed: int = 42) -> None:
        """
        Constructor of the class.

        Args:
            seed: Random seed for reproducibility.
        """

        # Set random seeds for reproducibility
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader, self.test_loader = self._get_loaders()
        self.model = YOLO().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = YOLOLoss()
        self.losses: dict[str, list[float | tuple[int, float]]] = {
            "coord": [],
            "obj": [],
            "noobj": [],
            "class": [],
            "loss": [],
            "map_train": [],
            "map_test": [],
        }

    def _get_loaders(self) -> tuple[DataLoader, DataLoader]:
        """
        Obtains the train and test loaders.

        Returns:
            Train and test loader.
        """

        train_dataset: RaccoonDataset = RaccoonDataset(
            "data/images_resized/train", "data/labels_resized/train"
        )
        test_dataset: RaccoonDataset = RaccoonDataset(
            "data/images_resized/test", "data/labels_resized/test"
        )

        train_dataloader: DataLoader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True
        )
        test_dataloader: DataLoader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True
        )

        return train_dataloader, test_dataloader

    def _append_losses(
        self,
        coord_loss: torch.Tensor,
        obj_loss: torch.Tensor,
        noobj_loss: torch.Tensor,
        class_loss: torch.Tensor,
        loss: torch.Tensor,
    ) -> None:
        """
        Appends the losses of the batch to the dictionary of the losses.

        Args:
            coord_loss: Loss for the predicted coordinates.
            obj_loss: Loss for the prediction of if an object exists in the cell.
            noobj_loss: Loss for the prediction of if an object does not exist in the
                cell.
            class_loss: Loss for the prediction of the class of the object.
            loss: Total loss.
        """

        self.losses["coord"].append(float(coord_loss.detach()))
        self.losses["obj"].append(float(obj_loss.detach()))
        self.losses["noobj"].append(float(noobj_loss.detach()))
        self.losses["class"].append(float(class_loss.detach()))
        self.losses["loss"].append(float(loss.detach()))

    def _make_epoch_train(self) -> None:
        """
        Performs one epoch of the training.
        """

        self.model.train()
        self.model.to(self.device)

        for images, targets in tqdm(self.train_loader, desc="Training"):
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            predictions: torch.Tensor = self.model(images)

            coord_loss: torch.Tensor
            obj_loss: torch.Tensor
            noobj_loss: torch.Tensor
            class_loss: torch.Tensor
            yolo_loss: torch.Tensor

            coord_loss, obj_loss, noobj_loss, class_loss, yolo_loss = self.criterion(
                predictions, targets
            )

            yolo_loss.backward()
            self.optimizer.step()

            self._append_losses(coord_loss, obj_loss, noobj_loss, class_loss, yolo_loss)

    def fit(
        self,
        n_epochs: int = EPOCHS,
        start_epochs: list[int] | None = None,
        save_every: int = 5,
        path_weights: str = PATH_MODEL,
        path_images: str = "images/training/evolution",
    ) -> None:
        """
        Training of the network. It saves the parameters of the trained model.

        Args:
            n_epochs: Number of epochs to train.
            start_epochs: List with the starting epoch for each graphic saved.
            save_every: To save the model and calculate mAP every 'save_every' epochs.
            path_weights: Path where the weights are saved.
            path_images: Path where we save the images of the evolution of the training.
        """

        os.makedirs(os.path.dirname(path_weights), exist_ok=True)

        for epoch in range(n_epochs):

            self._make_epoch_train()

            # Model saving through epochs
            if (epoch + 1) % save_every == 0:
                torch.save(self.model.state_dict(), path_weights)

                self.model.eval()
                self.model.to(self.device)

                with torch.no_grad():

                    # Train mAP
                    train_map: float = mean_average_precision("train", self.model)
                    self.losses["map_train"].append((epoch + 1, train_map))

                    # Test mAP
                    test_map: float = mean_average_precision("test", self.model)
                    self.losses["map_test"].append((epoch + 1, test_map))

                print(f"Epoch {epoch + 1}/{n_epochs}")
                print(f"Train mAP: {train_map:.4f}, Test mAP: {test_map:.4f}")

                # Set model back to train mode for next epoch
                self.model.train()

        # Save final model
        torch.save(self.model.state_dict(), path_weights)

        # Always plot training evolution - ensure directory exists
        img_dir = os.path.dirname(path_images)
        if img_dir:
            os.makedirs(img_dir, exist_ok=True)
        self._plot_training_evolution(start_epochs, path_images)

    def test(self) -> float:
        """
        Test of the trained network.

        Returns:
            Loss in the test set.
        """

        # Load trained model
        if os.path.exists(PATH_MODEL):
            self.model.load_state_dict(torch.load(PATH_MODEL, map_location=self.device))
            print(f"Loaded trained model from {PATH_MODEL}")

        else:
            print(f"No trained model found at {PATH_MODEL}")
            return 0.0

        self.model.eval()
        test_loss: float = 0.0
        total_batches: int = 0

        with torch.no_grad():
            for images, targets in tqdm(self.test_loader, desc="Testing"):

                images = images.to(self.device)
                targets = targets.to(self.device)

                predictions = self.model(images)

                _, _, _, _, batch_loss = self.criterion(predictions, targets)

                test_loss += float(batch_loss)
                total_batches += 1

        return test_loss / total_batches

    def _save_graphic_loss(self, start_epoch: int, path_images: str) -> None:
        """
        Saves a graphic starting from a specific epoch.

        Args:
            start_epoch: First epoch of the graphic.
            path_images: Path where the images are saved.
        """

        loss_data = self.losses["loss"][start_epoch:]
        x = range(start_epoch, start_epoch + len(loss_data))

        fig, axs = plt.subplots()
        axs.plot(x, self.losses["coord"][start_epoch:], label="Coordinates")
        axs.plot(x, self.losses["obj"][start_epoch:], label="Objects")
        axs.plot(x, self.losses["noobj"][start_epoch:], label="No Objects")
        axs.plot(x, self.losses["class"][start_epoch:], label="Classification")
        axs.plot(x, self.losses["loss"][start_epoch:], label="Total")

        axs.grid()
        axs.legend()
        axs.set_xlabel("Batch")
        axs.set_ylabel("Loss")
        axs.set_title("Evolution of the Loss During the Training")

        fig.savefig(f"{path_images}_loss_{start_epoch}.png")
        plt.close(fig)

    def _save_graphic_map(self, path_images: str) -> None:
        """
        Saves a graphic starting from a specific epoch.

        Args:
            path_images: Path where the images are saved.
        """

        x = [xx[0] for xx in self.losses["map_train"]]  # type: ignore

        fig, axs = plt.subplots()
        axs.plot(
            x, [xx[1] for xx in self.losses["map_train"]], label="Train"  # type: ignore
        )
        axs.plot(
            x, [xx[1] for xx in self.losses["map_test"]], label="Test"  # type: ignore
        )

        axs.grid()
        axs.legend()
        axs.set_xlabel("Epoch")
        axs.set_ylabel("mAP")
        axs.set_title("Evolution of the mAP During the Training")

        fig.savefig(f"{path_images}_map.png")
        plt.close(fig)

    def _plot_training_evolution(
        self,
        start_epochs: list[int] | None = None,
        path_images: str = "images/training/evolution",
    ) -> None:
        """
        Plots the evolution of the training loss.

        Args:
            start_epochs: List with the starting epoch for each graphic saved.
            path_images: Path where the images are saved.
        """

        if start_epochs is None:
            total_epochs = len(self.losses["loss"])
            if total_epochs < 50:
                start_epochs = [0]
            elif total_epochs < 100:
                start_epochs = [0, 50]
            else:
                start_epochs = [0, 50, 100]

        # Ensure directory exists before saving images
        img_dir = os.path.dirname(path_images)
        if img_dir:
            os.makedirs(img_dir, exist_ok=True)

        for start_epoch in start_epochs:
            self._save_graphic_loss(start_epoch, path_images=path_images)

        self._save_graphic_map(path_images)


if __name__ == "__main__":
    trainer = Trainer()
    trainer.fit()
