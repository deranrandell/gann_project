# Generative Adversarial Network (GAN) for Image Generation

A simple implementation of a Generative Adversarial Network (GAN) in PyTorch. This project generates images from random latent vectors using the CelebA dataset.

---

## Requirements

Before you start, ensure you have the following:

- Python 3.x
- `torch`, `torchvision`, `tqdm`, `numpy`, `matplotlib`

To install the required dependencies:

```bash
pip install torch torchvision tqdm numpy matplotlib

Clone the repository:
cd gann_project
```
## Download the CelebA dataset:
Place the dataset in the ./data directory. The script will automatically load the dataset from there.

To start training the GAN, run the following command:

```
python training/train.py

```
## Hyperparameters
You can modify these hyperparameters in train.py:
```
BATCH_SIZE: The number of images per batch (default: 64)
LATENT_DIM: The size of the latent vector (default: 100)
EPOCHS: The number of training epochs (default: 100)
LEARNING_RATE: Learning rate for Adam optimizers (default: 0.0002)
BETA1: Beta1 value for the Adam optimizer (default: 0.5)
IMAGE_SIZE: The size of the generated images (default: 64x64)
DEVICE: The device for training (CUDA or CPU)
```

## Training Process

The Discriminator is trained to differentiate between real and generated images.
The Generator learns to create images that appear as realistic as possible.
At the end of each epoch, generated images are saved to the generated_images/ directory.

Generated images are saved every epoch in the generated_images/ folder.

Example:

```
generated_images/
    epoch_0.png
    epoch_1.png
    epoch_2.png
    ...
```
## License
This project is licensed under the MIT License. See the LICENSE file for more details.

