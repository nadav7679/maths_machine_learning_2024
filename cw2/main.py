from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch.optim import Adam
from torch import nn
from datasets import load_dataset
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

from diffusion import Diffusion

BATCH_SIZE=64

# load dataset from the hub
dataset = load_dataset("fashion_mnist")
channels = 1

# define image transformations (e.g. using torchvision)
transform = Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
])


def transforms(examples):
    examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
    del examples["image"]
    return examples

transformed_dataset = dataset.with_transform(transforms).remove_columns("label")

dataloader = DataLoader(transformed_dataset['train'], batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# TODO: experiment with different betas and hyperparameters for noise addition.


# Main training loop
def train(diffusion: Diffusion, nr_epochs, optimizer, scheduler, device, dataloader=dataloader):
    for epoch in range(nr_epochs):
        # iterate through batches
        for i, data in enumerate(dataloader, 0):

            # get inputs
            images = data["pixel_values"]
            images = images.to(device)
            batch_size = images.shape[0]

            optimizer.zero_grad()

            # Get loss (using forward diffusion and run through unet)
            t = torch.randint(0, diffusion.T, (batch_size, ), device=device)
            loss = diffusion.get_loss(images, t)

            loss.backward()
            optimizer.step()



        # print results for last batch
        print(f"Epoch: {epoch + 1:03} | Loss: {loss} | lr: {optimizer.param_groups[0]['lr']}")
        # Update learning_rate
        scheduler.step()

    print('Finished Training')


# Run training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
diffusion = Diffusion(500, device)

print(f"Number of parameters: {sum(p.numel() for p in diffusion.unet.parameters() if p.requires_grad)}")

optimizer = torch.optim.Adam(diffusion.unet.parameters(), 0.00002)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

train(diffusion, 30, optimizer, scheduler, device)
torch.save(diffusion.unet, "unet_T500_C6_E30.tr")
