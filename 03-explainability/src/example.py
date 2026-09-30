"""
Script to generate some sample images for each method.
"""

import sys
from io import BytesIO
from pathlib import Path

import requests  # type: ignore[import-untyped]
import torch
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

# Add parent directory to path to allow running this script directly
if __name__ == "__main__":
    parent_dir = Path(__file__).resolve().parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

from src.visualize import Visualizer


def _obtain_images() -> tuple[torch.Tensor, torch.Tensor]:
    """
    Obtains a normalized and unnormalized image to plot.

    Returns:
        Normalized and unnormalized image. Dimensions: [channels, height, width].
    """

    # Original image
    img_url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    response = requests.get(img_url, timeout=60)
    original_image = Image.open(BytesIO(response.content)).convert("RGB")

    # Normalized image
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )  # ImageNet stats
    normalized_image = transform(original_image)

    # Unnormalized image
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    unnorm_image = transform(original_image)

    return normalized_image, unnorm_image


def main() -> None:
    """
    Generates one image for each method and removes the top pixels in ascending and
    descending order. All images will be saved in the 'images' folder.
    """

    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    norm_image, unnorm_image = _obtain_images()

    for method in [
        "occlusion",
        "guided_backprop",
        "integrated_gradients",
    ]:  # , "guided_backprop", "integrated_gradients"
        visualizer = Visualizer(method, model, norm_image, unnorm_image)  # type: ignore

        output_path = f"images/{visualizer.explainer.__class__.__name__}.png"
        visualizer.explain_with_plot(path=output_path)

        for remove_top in [True, False]:
            visualizer.remove_pixels(
                remove_top=remove_top,
                percentage=0.2,
                save_figs=True,
            )


if __name__ == "__main__":
    main()
