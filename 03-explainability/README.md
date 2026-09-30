# 03 Explainability

## Goal

Implement three attribution methods for an image classifier and check them by removing pixels in the order the method ranks them. The model is a torchvision ResNet50 with ImageNet weights and the example image is `dog.jpg` from the PyTorch Hub repository.

## What is implemented

- `src/utils.py`: `saliency_map`, the gradient of the class score with respect to the input, taking the maximum of the absolute value over channels.
- `src/explainers/occlusion.py`: `Occlusion` slides a rectangular mask over the image and records the change in the target-class score. The default mask is one tenth of the image width and height and the default stride is half the mask.
- `src/explainers/integrated_gradients.py`: `IntegratedGradients` accumulates gradients along the straight path from a baseline (zeros or Gaussian noise) to the input, with a configurable number of steps (default 10).
- `src/explainers/guided_backprop.py`: `GuidedBackprop` registers forward and backward hooks so that only positive gradients pass back through the ReLU layers, and removes the hooks afterwards.
- `src/visualize.py`: `Visualizer` picks an explainer, plots the explanation and, in `remove_pixels`, sets the highest-ranked ("top") or lowest-ranked ("worst") pixels to zero in steps of a fixed percentage and saves the images.
- `src/example.py`: runs the three methods on the example image.

## Results

`images/` holds one explanation per method (`Occlusion.png`, `IntegratedGradients.png`, `GuidedBackprop.png`) and, under `images/remove/<method>/{top,worst}/`, the image with 0, 20, 40, 60, 80 and 100 percent of the pixels removed. No quantitative results are stored in the repository.

## Running it

`python -m src.example` downloads the image and writes the figures to `images/`. The course notes that Occlusion takes several minutes. The tests and the reference images supplied with the assignment are not included. Dependencies are in `pyproject.toml`.
