# 01 Transfer learning across modalities

## Goal

Build a small system that takes an ImageNet-pretrained network and adapts it to a different dataset, with three strategies for how much of the network is trained. The datasets are CIFAR-10, the Oxford-IIIT Pet dataset (reduced to cat versus dog) and PatchCamelyon (PCAM, histopathology patches).

## What is implemented

- `src/data_manager.py`: `DataManager`, a registry of the three datasets with their class counts and splits, ImageNet-compatible transforms (resize, tensor conversion, normalisation) and train and validation `DataLoader`s.
- `src/model_handler.py`: `ModelHandler` loads ResNet18, VGG16, AlexNet or Inception v3 with torchvision ImageNet weights and replaces the final classification layer to match the number of classes of the target dataset.
- `src/strategies.py`: a `FineTuningStrategy` base class and three strategies.
  - `LastLayerStrategy` freezes every parameter except the final layer.
  - `FullFineTuningStrategy` trains all parameters.
  - `FeatureExtractionStrategy` replaces the final layer with `nn.Identity`, extracts features for the train and validation sets and fits a logistic regression on them.
- `src/visualizer.py`: `FeatureVisualizer` extracts features with the feature-extraction strategy, reduces them to two dimensions with PCA and plots them by class. It is used to compare how well ImageNet features separate cats and dogs with how they separate PCAM patches.

## Results

The repository contains no result logs or figures for this project, so none are reported.

## Running it

The code expects the course template around it. `src/utils.py`, with `get_final_layer` and `set_nested_attr`, is imported by `model_handler.py` and `strategies.py`. It was supplied with the assignment and is not included. Dependencies are listed in `pyproject.toml` (Python 3.13, torch, torchvision, scikit-learn, matplotlib). PCAM has to be downloaded manually.
