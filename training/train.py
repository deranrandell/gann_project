import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import save_image

def train_gan(generator, discriminator, dataloader, optimizer_g, optimizer_d, epochs, latent_dim, device):
    for epoch in range(epochs):
        for i, (real_images, _) in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)

            # Train Discriminator
            optimizer_d.zero_grad()
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_images = generator(z)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            real_loss = F.binary_cross_entropy(discriminator(real_images), real_labels)
            fake_loss = F.binary_cross_entropy(discriminator(fake_images.detach()), fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            g_loss = F.binary_cross_entropy(discriminator(fake_images), real_labels)
            g_loss.backward()
            optimizer_g.step()

            if i % 100 == 0:
                print(f'Epoch [{epoch}/{epochs}], Step [{i}/{len(dataloader)}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')

        # Save Images at each epoch
        save_generated_images(fake_images, epoch)

def save_generated_images(images, epoch, path='generated_images/'):
    save_image(images, f'{path}/epoch_{epoch}.png', normalize=True)
