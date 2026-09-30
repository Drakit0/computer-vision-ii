# 02 YOLO object detection

## Goal

Implement a YOLO (version 1) detector and train it from scratch on a raccoon dataset. The output is a grid of S x S cells, each predicting B bounding boxes and C class scores.

## What is implemented

Settings are in `src/constants.py`: S = 7, B = 2, C = 1, 400 x 400 inputs, lambda_coord = 5, lambda_noobj = 0.25, Adam with learning rate 1e-3, batch size 16, 45 epochs.

- `src/data/dataset.py`: `RaccoonDataset` reads an image and its text label file and builds the S x S x (5B + C) target tensor.
- `src/model/model.py`: `YOLO`. The convolutional part is built from a list in `constants.py` of (kernel, channels, stride, padding) tuples and max pooling marks, each convolution followed by batch normalisation and LeakyReLU(0.1). A multilayer perceptron (lazy linear 1024, LeakyReLU, dropout 0.3, linear) outputs the grid. `predict` and `draw_predictions` apply non-maximum suppression and draw the boxes.
- `src/model/loss.py`: `YOLOLoss` with the four terms of the original paper: box coordinates (square root of width and height, sign kept for negative predictions), object confidence, no-object confidence and class. The responsible box in each cell is the one with the higher IoU against the target.
- `src/utils.py`: IoU, non-maximum suppression, conversion from cell-relative to image coordinates, and mean average precision.
- `src/model/train.py`: `Trainer` with the train loop, mAP on train and test every 5 epochs, checkpointing and plots.
- `src/example.py`: draws predictions for chosen images.

## Results

`images/training/evolution_map.png` plots train and test mAP every 5 epochs. The curves are not monotonic. Reading the plot, test mAP drops to about 0.09 at epoch 25, peaks near 0.64 at epoch 30 and ends at about 0.55 at epoch 45, while train mAP ends near 0.72. The loss curves are in `images/training/evolution_loss_*.png`. Predictions on train and test images are in `images/predictions/` (red boxes are predictions).

## Running it

Dataset preparation (`src/data/transform_data.py`), the raccoon images and labels, the tests and the 55 MB weight file are not included. With the data prepared by the course script in `data/`, training is `python -m src.model.train` and predictions are `python -m src.example`. Dependencies are in `pyproject.toml`.
