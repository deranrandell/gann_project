import os
from torchvision.utils import save_image

def save_generated_images(images, epoch, path='generated_images'):
    os.makedirs(path, exist_ok=True)
    save_image(images, f'{path}/epoch_{epoch}.png', normalize=True)
