# 04 Adversarial attacks

## Goal

Attack a small CIFAR-10 classifier (the course describes it as reaching about 70 percent accuracy) with one white-box and one black-box method, and implement CutMix augmentation.

## What is implemented

- `src/fgsm.py`: `FastGradientSignMethod`, the Fast Gradient Sign Method of Goodfellow et al. The perturbation is epsilon times the sign of the gradient of the loss with respect to the input, added to the image.
- `src/opa.py`: `OnePixelAttack`, the black-box attack of Su et al. using differential evolution. Each candidate is a tuple (x, y, R, G, B) in [0, 1]. A new candidate is built as p_r1 + F (p_r2 - p_r3) with F = 0.5 from distinct random individuals, clipped to range, and it replaces its parent only if the softmax probability of the true class goes down. The population size defaults to 100. The attack can change several pixels and stops early when the predicted label changes.
- `src/cutmix.py`: `cutmix` samples lambda from a Beta distribution, builds a rectangle whose area depends on 1 - lambda, pastes it from the second image into the first and mixes the labels with the same weights.

## Results

`images/` has the figures my runs produced: `FGSM_0.01.png`, `FGSM_0.1.png` and `FGSM_1.png` (epsilon 0.01, 0.1 and 1), `OPA_1.png` to `OPA_3.png`, and `CutMix_2.png` and `CutMix_5.png`. No accuracy or success-rate numbers are stored in the repository.

## Running it

The model (`src/models/`, with its weights), `src/utils.py` (visualisation helpers and seeding) and the tests were supplied with the assignment and are not included. The three files here import from them. Dependencies are in `pyproject.toml`.
