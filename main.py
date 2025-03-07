import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.generator import Generator
from models.discriminator import Discriminator
from training.train import train_gan
from utils.visualizations import save_generated_images
from torchvision import datasets, transforms

# Hyperparameters
BATCH_SIZE = 64
LATENT_DIM = 100
EPOCHS = 100
LEARNING_RATE = 0.0002
BETA1 = 0.5
IMAGE_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data Loader
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = datasets.CelebA(root='./data', download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Models
generator = Generator(LATENT_DIM).to(DEVICE)
discriminator = Discriminator().to(DEVICE)

# Optimizers
optimizer_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

# Training the GAN
train_gan(generator, discriminator, dataloader, optimizer_g, optimizer_d, EPOCHS, LATENT_DIM, DEVICE)
