import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from unet import SimpleUnet


def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class Diffusion():
    def __init__(self, T: int, base_channels, device):
        self.unet = SimpleUnet(base_channels, device).to(device=device)
        self.T = T
        self.device = device

        #: A list of T equidistant points between 0.0001 and 0.02
        self.beta = torch.linspace(0.0001, 0.02, T, device=device)
        #: A list of T equidistant points compliment to self.beta
        self.alpha = 1 - self.beta

        #: A list of T alpha_bar values. Each entry is cumprod of alphas up to time t
        self.alpha_bar = torch.cumprod(self.alpha, 0)
        #: Precalculating values
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)
        alpha_bar_prev = torch.cat((torch.tensor([1.]).to(device=device), self.alpha_bar[:-1]), 0)
        self.posterior_variance = self.beta * (1 - alpha_bar_prev) / (1 - self.alpha_bar)

    def forward_diffusion_sample(self, x_0: torch.Tensor, t: torch.Tensor):
        """
        Forward diffusion process, i.e. take an image or batch of images and return the images at timestep t, which
        a certain amount of noise determined by t.
        :param x_0: torch.Tensor of clean datapoint images in batch of shape [batch_size, 1, 28, 28]
        :param t: torch.Tensor of values ranging from 0 to T with shape [batch_size]
        :param device: Device
        :return: torch.Tensor with shape as x_0 but with noise determined by t.
        """
        if x_0.shape[0] != t.shape[0]:
            raise ValueError("x_0 and t must have same shape[0] which is batch_size")

        pure_noise = torch.randn_like(x_0, dtype=torch.float32, device=self.device)
        scaled_noise = get_index_from_list(self.sqrt_one_minus_alpha_bar, t, x_0.shape) * pure_noise

        return get_index_from_list(self.sqrt_alpha_bar, t, x_0.shape) * x_0 + scaled_noise, pure_noise

    def get_loss(self, x_0, t):
        """
        Loss per batch (x_0, t).
        Sample forward diffusion, get noisy image, get noise from unet, compare noise from unet to image noise.

        :param x_0: torch.Tensor of clean datapoint images in batch of shape [batch_size, 1, 28, 28]
        :param t: torch.Tensor of values ranging from 0 to T with shape [batch_size]
        :return: MSE loss between the noise and model's prediction of noise, reduced over batch.
        """
        if x_0.shape[0] != t.shape[0]:
            raise ValueError("x_0 and t must have same shape[0] which is batch_size")

        xt, noise = self.forward_diffusion_sample(x_0, t)

        return nn.functional.mse_loss(noise, self.unet(xt, t))

    @torch.no_grad()
    def sample_timestep(self, xt, t, i):
        """
        Calls the model to predict the noise in the image based on t and returns the denoised image.
        Applies noise to this image, if we are not in the last step yet.

        :param xt: torch.Tensor of a *single* noisy image, shape [28, 28]
        :param t: torch.Tensor the timestep of the noisy image
        :param i: no idea
        :return:
        """
        if xt.shape[0] != t.shape[0] and xt.shape[0] != 1:
            raise ValueError("xt and t must have same shape[0] which is 1")

        beta = get_index_from_list(self.beta, t, xt.shape)
        sqrt_alpha_recip = torch.sqrt(1/(1-beta))
        sqrt_one_minus_alpha_bar = get_index_from_list(self.sqrt_one_minus_alpha_bar, t, xt.shape)
        posterior_variance = get_index_from_list(self.posterior_variance, t, xt.shape) * torch.sqrt(self.beta[i])

        mean = sqrt_alpha_recip * (xt - (beta/sqrt_one_minus_alpha_bar) * self.unet(xt, t))
        if i == 0:
            return mean

        return mean + torch.sqrt(posterior_variance) * torch.randn(xt.shape, device=xt.device)

    @torch.no_grad()
    def sample(self, shape):
        batch_size = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=self.device)
        imgs = []
        for i in tqdm(reversed(range(0, self.T)), desc='Sampling loop time step', total=self.T):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img = self.sample_timestep(img, t, i)
            imgs.append(img.cpu().numpy())

        return imgs



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion = Diffusion(10, device)

    # Sample from posterior (random noise -> image)
    imgs = diffusion.sample([4, 1, 28, 28])
    img_seq_0 = [imgs_t[0] for imgs_t in imgs]
    for i in reversed(range(10)):
        plt.imshow(img_seq_0[i][0])
        plt.pause(0.5)

    # test = torch.randn((28, 28), device=device).unsqueeze(0).unsqueeze(0)
    # t = torch.Tensor([3]).to(device=device)
    # diffusion.sample_timestep(test, t)
