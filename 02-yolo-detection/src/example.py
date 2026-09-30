"""
This script contains the code for making some predictions and save the images.
"""

from typing import Literal
import os
import sys

# Add project root to path when running as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from PIL import ImageDraw

from src.constants import PATH_MODEL, TARGET_SIZE_IMG
from src.model.model import YOLO


def _get_real_labels(index: int, subset: Literal["train", "test"]) -> list[list[float]]:
    """
    Obtains the labels associated to the image.

    Args:
        index: Index the image to predict.
        subset: Subset to which the image belongs to.

    Returns:
        Labels of the image in the correct format: (x1, y1, x2, y2). Dimensions:
        [n_labels, 4].
    """

    with open(f"data/labels_resized/{subset}/{index}.txt", encoding="utf8") as f:
        boxes = []
        for label in f.readlines():
            _, c_x, c_y, width, height = [
                float(x) if float(x) != int(float(x)) else int(x)
                for x in label.replace("\n", "").split()
            ]
            boxes.append(
                [
                    (c_x - width / 2) * TARGET_SIZE_IMG[0],
                    (c_y - height / 2) * TARGET_SIZE_IMG[1],
                    (c_x + width / 2) * TARGET_SIZE_IMG[0],
                    (c_y + height / 2) * TARGET_SIZE_IMG[1],
                ]
            )

    return boxes


def main(indexes: list[int], subset: Literal["train", "test"]) -> None:
    """
    Draws some predictions. If there were more classes, may be we could also draw at the
    top of each box its class.

    Args:
        indexes: Indexes of the images to predict.
        subset: Subset where the images reside.
    """

    model = YOLO()
    model.load_state_dict(torch.load(PATH_MODEL, weights_only=True))

    # Create output directory if it doesn't exist
    os.makedirs(f"images/predictions/{subset}", exist_ok=True)

    for index in indexes:
        img_path = f"data/images_resized/{subset}/raccoon-{index}.jpg"
        output_path = f"images/predictions/{subset}/raccoon-{index}.jpg"

        # Predictions
        img = model.draw_predictions(img_path)
        output_img = ImageDraw.Draw(img)

        # Targets
        real_boxes = _get_real_labels(index, subset)  # type: ignore
        for box in real_boxes:
            output_img.rectangle(box, outline="blue", width=3)

        img.save(output_path)


if __name__ == "__main__":
    N = 10
    INDEXES = list(range(N))
    for SUBSET in ["train", "test"]:
        main(INDEXES, SUBSET)  # type: ignore
