import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
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


class Diffusion:
    """
    Class for the diffusion model.
    """

    def __init__(self, T: int, base_channels, device, upsample=False, cosine=False, second_variance=True, dropout=None,
                 silu=False):
        """
        Initializes the diffusion model.

        Args:
            T (int): Number of timesteps.
            base_channels (int): Number of channels at first layer of UNet.
            device (torch.device): Device to run the model on.
            upsample (bool, optional): Whether to use nearest-neighbor upsample or convTranspose. Defaults to False.
            cosine (bool, optional): Use cosine beta schedule or linear schedule. Defaults False.
            second_variance (bool, optional): Whether to use the second type of variance. Defaults to True.
            dropout (float, optional): Dropout rate. Defaults to None.
            silu (bool, optional): Whether to use SiLU activation or ReLU. Defaults to False.
        """
        self.unet = SimpleUnet(base_channels, device, upsample, dropout, silu).to(device=device)
        self.T = T
        self.device = device

        #: beta_t, can be either linear or cosine
        self.beta = self.beta_schedule(T, cosine).to(device=device)
        #: A list of T equidistant points compliment to self.beta
        self.alpha = 1 - self.beta

        #: A list of T alpha_bar values. Each entry is cumprod of alphas up to time t
        self.alpha_bar = torch.cumprod(self.alpha, 0)
        #: Precalculating values
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)

        # Two possible options for posterior_variance
        if second_variance:
            alpha_bar_prev = torch.cat((torch.tensor([1.]).to(device=device), self.alpha_bar[:-1]), 0)
            self.posterior_variance = self.beta * (1 - alpha_bar_prev) / (1 - self.alpha_bar)
        else:
            self.posterior_variance = self.beta.clone()


    @staticmethod
    def beta_schedule(T, cosine, start=0.0001, end=0.02):
        """
        Generate the beta_t for the noisy timesteps.

        Args:
            T (int): Number of timesteps.
            cosine (bool): Whether to use cosine nosise, based on reference [4].
            start (float): Start value of the linear schedule.
            end (float): End value of the linear schedule.

        Returns:
            torch.Tensor: Beta schedule tensor.
        """
        if not cosine:
            return torch.linspace(start, end, T)

        alpha_bar = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        betas = []
        for i in range(T):
            t1 = i / T
            t2 = (i + 1) / T
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))

        return torch.tensor(betas)

    def forward_diffusion_sample(self, x_0: torch.Tensor, t: torch.Tensor):
        """
        Forward diffusion process.

        Args:
            x_0 (torch.Tensor): Clean datapoint images in batch.
            t (torch.Tensor): Timestep tensor.

        Returns:
            tuple: Noisy image and pure noise tensor.
        """
        if x_0.shape[0] != t.shape[0]:
            raise ValueError("x_0 and t must have same shape[0] which is batch_size")

        pure_noise = torch.randn_like(x_0, dtype=torch.float32, device=self.device)
        scaled_noise = get_index_from_list(self.sqrt_one_minus_alpha_bar, t, x_0.shape) * pure_noise
        xt = get_index_from_list(self.sqrt_alpha_bar, t, x_0.shape) * x_0

        return torch.clamp(xt + scaled_noise, -1.0, 1.0), pure_noise  # Clamp to stay in [-1,1] range.

    def get_loss(self, x_0, t):
        """
        Loss per batch.

        Args:
            x_0 (torch.Tensor): Clean datapoint images in batch.
            t (torch.Tensor): Timestep tensor.

        Returns:
            torch.Tensor: MSE loss between the noise and model's prediction of noise.
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

        Args:
            xt (torch.Tensor): Noisy image tensor.
            t (torch.Tensor): Timestep tensor.
            i (int): Timestep index.

        Returns:
            torch.Tensor: Denoised image.
        """
        if xt.shape[0] != t.shape[0] and xt.shape[0] != 1:
            raise ValueError("xt and t must have same shape[0] which is 1")

        beta = get_index_from_list(self.beta, t, xt.shape)
        sqrt_alpha_recip = torch.sqrt(1 / (1 - beta))
        sqrt_one_minus_alpha_bar = get_index_from_list(self.sqrt_one_minus_alpha_bar, t, xt.shape)
        posterior_variance = get_index_from_list(self.posterior_variance, t, xt.shape)

        mean = sqrt_alpha_recip * (xt - (beta / sqrt_one_minus_alpha_bar) * self.unet(xt, t))
        if not i:
            return mean

        return mean + torch.sqrt(posterior_variance) * torch.randn(xt.shape, device=xt.device)

    @torch.no_grad()
    def sample(self, shape):
        """
        Sample from the diffusion model.

        Args:
            shape (list or tuple): Shape of the tensor to sample.

        Returns:
            list: List of sampled tensors.
        """
        batch_size = shape[0]
        img = torch.randn(shape, device=self.device)
        imgs = []
        for i in tqdm(reversed(range(0, self.T)), desc='Sampling loop time step', total=self.T):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img = self.sample_timestep(img, t, i)
            imgs.append(img.cpu().numpy())

        return imgs


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion = Diffusion(10, 32, device, False, True, silu=True)

    # Sample from posterior (random noise -> image)
    imgs = diffusion.sample([4, 1, 28, 28])
    img_seq_0 = [imgs_t[0] for imgs_t in imgs]
    for i in reversed(range(10)):
        plt.imshow(img_seq_0[i][0])
        plt.pause(0.5)
