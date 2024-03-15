import torch.nn.functional as F
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


IMG_SIZE = 28
BATCH_SIZE = 128

# load dataset from the hub
dataset = load_dataset("fashion_mnist")
channels = 1

# define image transformations (e.g. using torchvision)
transform = Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
])


# define function
def transforms(examples):
    examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
    del examples["image"]
    return examples


def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

transformed_dataset = dataset.with_transform(transforms).remove_columns("label")

dataloader = DataLoader(transformed_dataset['train'], batch_size=BATCH_SIZE, shuffle=True, drop_last=True)



def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    '''
    output a vector of size timesteps that is equally spaced between start and end; this will be the noise that is added in each time step.
    '''
    return torch.linspace(start, end, timesteps)

def forward_diffusion_sample(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device="cpu"):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    Hint: use the get_index_from_list function to select the right values at time t.
    """
    epsilon = torch.randn_like(x_0, dtype=torch.float32, device=device)
    # print(get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape), sqrt_alphas_cumprod[t[...]])
    noise = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape) * epsilon

    return get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape) * x_0 + noise, noise


# Define beta schedule
T = 300
betas = linear_beta_schedule(timesteps=T)

# Pre-calculate different terms for closed form
# ADD HERE THE COMPUTATIONS NEEDED for sqrt_alphas_cumprod and sqrt_one_minus_alphas_cumprod
alphas = 1 - betas
cumprod = torch.cumprod(alphas, 0)
sqrt_alphas_cumprod = torch.sqrt(cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - cumprod)

# Simulate forward diffusion
simulate_forward_diffusion = False
if simulate_forward_diffusion:
    batch = next(iter(dataloader))["pixel_values"]
    num_images = 10
    stepsize = int(T/num_images)

    for idx in range(0, T, stepsize):
        t = torch.Tensor([idx]).type(torch.int64)
        img, noise = forward_diffusion_sample(batch[0,:,:,:], t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        plt.imshow(img.reshape(28, 28), cmap="gray")
        plt.show()

# TODO: experiment with different betas and hyperparameters for noise addition.


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        res = torch.zeros((len(time), self.dim), dtype=torch.float32)

        pos = torch.arange(len(time)).unsqueeze(1)
        two_i = torch.arange(0, self.dim, 2)

        arg = torch.exp(torch.log(pos) - two_i * torch.log(torch.tensor(10**4)) / self.dim)

        res[:, 0::2] = torch.sin(arg)
        res[:, 1::2] = torch.cos(arg)

        return res

show_sin_embed = False
if show_sin_embed:
    d = 100
    N = 60
    embed = SinusoidalPositionEmbeddings(d)
    res = embed(torch.linspace(0.001, 1, N))

    plt.imshow(res)
    plt.xlabel(f"Embedding dimension - d={d}")
    plt.ylabel(f"Original dimension - time position - N={N}")
    plt.gca().invert_yaxis()
    plt.show()

# TODO: add discussion and nice plot of embedding
