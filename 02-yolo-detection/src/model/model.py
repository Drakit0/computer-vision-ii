"""
Script to define the architecture for the model.
"""

import torch
from PIL import Image, ImageDraw
from PIL.JpegImagePlugin import JpegImageFile
from torch import nn
from torchvision import transforms

from src.constants import MODEL, TARGET_SIZE_IMG, B, C, S
from src.utils import nms, to_image_coords


class CNNBlock(nn.Module):
    """
    Class to build a CNN block consisting of Conv + Batch Norm + LeakyReLU.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        """
        Constructor of the class.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            kernel_size: Kernel size.
            stride: Stride.
            padding: Padding.
        """

        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor. Dimensions: [batch, in_channels, height, width].

        Returns:
            Output tensor. Dimensions: [batch, out_channels, h_out, w_out] (h_out and
            w_out can be checked here:
            https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html).
        """

        return self.leaky_relu(self.batchnorm(self.conv(x)))


class YOLO(nn.Module):
    """
    Class to build the YOLO architecture.
    """

    def __init__(self, in_channels: int = 3) -> None:
        """
        Constructor of the class.

        Args:
            in_channels: First input channels for the convolutional blocks.
        """

        super().__init__()

        self.cnn = self._create_cnn(MODEL, in_channels)
        self.mlp = self._create_mlp()

    def _create_cnn(
        self, architecture: list[tuple[int, int, int, int] | str], in_channels: int
    ) -> nn.Module:
        """
        Creates the convolutional layers.

        Args:
            architecture: Architecture of the convolutional layers. It is explained
                in the constants.py file.
            in_channels: Input channels for the first convolution.

        Returns:
            Architecture of the convolutional layers.
        """

        cnn_model: list[CNNBlock | nn.MaxPool2d] = []

        kernel_size: int = 0
        stride: int = 0
        padding: int = 0

        for layer in architecture:

            if isinstance(layer, tuple):
                kernel_size = layer[0]
                stride = layer[2]
                padding = layer[3]

                cnn_model.append(
                    CNNBlock(
                        in_channels,
                        layer[1],
                        kernel_size,
                        stride,
                        padding,
                    )
                )

                in_channels = layer[1]

            else:
                cnn_model.append(nn.MaxPool2d((2, 2), (2, 2)))

        return nn.Sequential(*cnn_model)

    def _create_mlp(self) -> nn.Module:
        """
        Creates the fully connected layers. You have the freedom to define the number
        and type of layers (for example, linear, dropout, leaky relu...) as you wish.

        Returns:
            Architecture of the fully connected layers.
        """

        mlp_model: nn.Sequential = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(1024, S * S * (5 * B + C)),
        )

        return mlp_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor. Dimensions: [batch, channels, height, width].

        Returns:
            Output tensor: Dimensions: [batch, S, S, 5 * B + C].
        """

        return self.mlp(self.cnn(x)).view(-1, S, S, 5 * B + C)

    def predict(self, img_path: str) -> list[torch.Tensor]:
        """
        Prediction given an image.

        Args:
            img_path: Path to the image (jpg, png...).

        Returns:
            Bounding boxes after NMS. Each one has dimension C + 5 and the last 4
            elements indicate the coordinates of the box.
        """

        self.eval()

        x = transforms.ToTensor()(
            Image.open(img_path).resize(TARGET_SIZE_IMG).convert("RGB")
        )

        # Move input to the same device as the model
        device = next(self.parameters()).device
        x = x.to(device)

        predicted_boxes = self(x.unsqueeze(0)).squeeze(0)
        scaled_boxes = to_image_coords(
            predicted_boxes, TARGET_SIZE_IMG[0], TARGET_SIZE_IMG[1]
        )

        return nms(scaled_boxes)

    def draw_predictions(self, img_path: str, output_path: str = "") -> JpegImageFile:
        """
        Draws the predicted bounding boxes of the image.

        Args:
            img_path: Path to the image.
            output_path: Path were the image created is saved (only if a path is given).

        Returns:
            Image with the bounding boxes.
        """

        img = Image.open(img_path).resize(TARGET_SIZE_IMG).convert("RGB")
        output_img = ImageDraw.Draw(img)
        predicted_boxes = self.predict(img_path)

        for predicted_box in predicted_boxes:
            c_x, c_y, width, height = predicted_box[C + 1 :]
            # Change coordinates to (x1, y1, x2, y2)
            try:
                output_img.rectangle(
                    [
                        float(c_x - width / 2),
                        float(c_y - height / 2),
                        float(c_x + width / 2),
                        float(c_y + height / 2),
                    ],
                    outline="red",
                    width=3,
                )
            except ValueError:  # wrong coordinates (for example, y1 < y0)
                pass

        if output_path:
            img.save(output_path)

        return img  # type: ignore
