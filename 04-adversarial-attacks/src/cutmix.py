import torch
import numpy as np


def cutmix(
    images: torch.Tensor, labels: tuple[int, int], alpha: float = 1.0
) -> tuple[torch.Tensor, float]:
    """
    Applies CutMix to a pair of images.

    Parameters
    ----------
    inputs : Pair of images. Dimensions: [2, channels, height, width].
    labels : Labels of both images.
    alpha  : Parameter for the beta distribution.

    Returns
    -------
    x_cutmix : Modified image. Dimensions: [channels, height, width].
    y_cutmix : Modified label.
    """

    lmbd: float = np.random.beta(alpha, alpha)

    H: int
    W: int
    _, _, H, W = images.shape

    r_x: int = int(np.random.uniform(0, W))
    r_y: int = int(np.random.uniform(0, H))
    r_w: int = int(W * np.sqrt(1 - lmbd))
    r_h: int = int(H * np.sqrt(1 - lmbd))

    mask: torch.Tensor = torch.zeros_like(images[0, ...].squeeze(0))
    mask[:, r_y:r_y + r_h, r_x:r_x + r_w] = 1

    img_0 = mask * images[0, ...].squeeze(0)
    img_1 = (1 - mask) * images[1, ...].squeeze(0)
    x_cut_mix: torch.Tensor = img_0 + img_1
    y_cut_mix: float = lmbd * labels[0] + (1 - lmbd) * labels[1]

    return x_cut_mix, y_cut_mix