"""
Creates a Pytorch dataset to load the dataset used.
"""

import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.constants import B, C, S


class RaccoonDataset(Dataset):
    """
    Adapted Dataset class for our dataset.
    """

    def __init__(
        self,
        img_dir: str,
        label_dir: str,
    ) -> None:
        """
        Constructor of the class.

        Args:
            img_dir: Path to the directory of the images.
            label_dir: Path to the directory of the labels.
        """

        self.imgs_dir = img_dir
        self.labels_dir = label_dir

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            Length of the dataset.
        """

        return len(os.listdir(self.labels_dir))

    def _generate_label(self, boxes: list[list[int | float]]) -> torch.Tensor:
        """
        Generates the labels with coordinates respect to the cells, not the hole image.
        We will assume at most we can have one object per cell. Therefore, the
        dimensions of the generated label will be [S, S, 5 * B + C]. If we wanted to
        detect at most n objects the dimensions would be [S, S, n, 5 * B + C].

        Args:
            boxes: Labels with respect to the hole image. Each label has 5 elements:
                class, center_x, center_y, width and height.

        Returns:
            Labels with respect to the cells. Dimensions: [S, S, 5 * B + C].
        """

        labels: torch.Tensor = torch.zeros((S, S, 5 * B + C))

        for img_class, OX, OY, W, H in boxes:
            i: int = int(OX * S)
            j: int = int(OY * S)

            if labels[i, j, int(img_class)] == 0:

                # Coordinates to the cell
                x_cell: float = OX * S - i
                y_cell: float = OY * S - j
                w_cell: float = W * S
                h_cell: float = H * S

                labels[i, j, C : C + 5] = torch.tensor(
                    [1, x_cell, y_cell, w_cell, h_cell]
                )
                labels[i, j, int(img_class)] = 1

        return labels

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the image and its labels.

        Args:
            index: Index of the image and labels.

        Returns:
            The image and its labels. Dimensions of the image: [channels, height,
            width]. Dimensions of the labels: [S, S, 5 * C + B].
        """

        # Load image - using index to match with label filename
        img_path: str = os.path.join(self.imgs_dir, f"raccoon-{index}.jpg")

        pil_image: Image.Image = Image.open(img_path).convert("RGB")
        transform: transforms.Compose = transforms.Compose(
            [transforms.Resize((400, 400)), transforms.ToTensor()]
        )
        image: torch.Tensor = transform(pil_image)

        # Load labels

        with open(os.path.join(self.labels_dir, f"{index}.txt"), "r") as f:
            line: str = str(f.readline())
            line_split: list[str] = line.split(" ")
            labels: torch.Tensor = self._generate_label(
                [[float(i) for i in line_split]]
            )

        return image, labels
