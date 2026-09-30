# Computer Vision II

Coursework from Computer Vision II (Comillas ICAI, autumn 2025): five lab projects and three paper summaries by Pablo Tuñón Laguna. The code is PyTorch.

## Projects

| Folder | Topic |
|---|---|
| `01-transfer-learning/` | Transfer learning across image modalities (CIFAR-10, Oxford-IIIT Pet, PatchCamelyon) with three fine-tuning strategies and a PCA feature plot |
| `02-yolo-detection/` | YOLO (version 1 style) object detector trained from scratch to find raccoons |
| `03-explainability/` | Occlusion, Integrated Gradients and Guided Backpropagation, plus a pixel-removal check |
| `04-adversarial-attacks/` | FGSM, One Pixel Attack with differential evolution, and CutMix |
| `05-generative-models/` | A DCGAN on MNIST and a VQ-VAE with an EMA codebook on CIFAR-10 |
| `paper-notes/` | My written summaries of the Inception, YOLO and Gaze-LLE papers (LaTeX source and PDF) |

Each folder has its own README with what was implemented and how to run it.

## What is not here

The assignments were built on templates provided by the course. The assignment scaffolding (templates, statements, automated tests, helper modules, datasets and the course papers) belongs to the course and is not included. Only the code I wrote or completed, the figures my own runs produced and my own notes are in this repository. Some files therefore import helper modules that are not present, and each project README says which. The repository is a record of the work and does not run without the course template.

There is no licence file for the same reason.
