# 05 Generative models

The fifth lab had two sessions: a GAN on MNIST and a VQ-VAE on CIFAR-10. Each has its own folder.

## gan/

A deep convolutional GAN on MNIST (28 x 28, one channel).

- `src/model.py`
  - `Discriminator`: three convolution blocks (kernels 4, 4 and 3, stride 2, batch normalisation except in the first, LeakyReLU) followed by a linear layer that outputs one logit.
  - `Generator`: linear layer, batch normalisation and ReLU to a 128 x 7 x 7 map, two transposed-convolution blocks up to 28 x 28 and a final transposed convolution with a sigmoid output.
  - `gan_loss`: binary cross entropy with logits against all-ones or all-zeros targets, plus `get_disc_loss` and `get_gen_loss`.
- `src/train.py`: `Trainer` alternates a discriminator step and a generator step on each batch. Defaults are Adam (learning rate 0.0002, betas 0.5 and 0.999), latent size 64, a 90/10 train/validation split, batch size 128 and 200 epochs. Samples from a fixed noise batch are saved each epoch to follow the generator.
- `src/utils.py`: seeding, noise sampling and plotting helpers.
- Results: `images/gan_losses.png` (loss curves) and `images/generated_samples_per_epoch.png`. Trained weights are in `weights/`. No numeric metric is stored.

Run with `python -m src.train` from `gan/`. MNIST is downloaded by torchvision. The tests are not included.

## vqvae/

A Vector Quantised VAE on CIFAR-10 following van den Oord et al.

- `src/networks.py`: residual block, residual stack, `Encoder` and `Decoder` (4x downsampling and upsampling, 32 x 32 to 8 x 8 and back).
- `src/vq_vae.py`: `VectorQuantizer` (codebook loss, commitment loss, straight-through gradient), `VectorQuantizerEMA` (codebook updated by exponential moving averages) and `VQVAE`, which joins the encoder, a 1 x 1 pre-quantisation convolution, the quantiser and the decoder.
- `src/ema_update.py`, `src/loss.py`: the EMA update and the loss terms, with the reconstruction error normalised by the data variance.
- `src/config.py`, `src/train.py`, `src/training.py`: settings (512 embeddings of dimension 64, commitment cost 0.25, decay 0.99, 15000 updates, batch size 256, learning rate 1e-3, mixed precision) and the training loop.
- `src/checkpoint_utils.py`, `src/utils.py`: loading checkpoints, reconstructions, codebook sampling and plots.
- Result: `checkpoint_reconstructions.png` shows 16 CIFAR-10 test images and their reconstructions after loading the course's template checkpoint into this implementation. Its title reports a VQ loss of 0.029 and a perplexity of 301.8.

The template checkpoint, the sample images supplied by the course, the test suite and the data are not included. CIFAR-10 is downloaded by torchvision.
