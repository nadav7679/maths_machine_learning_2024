import math

import torch
import matplotlib.pyplot as plt


class SinusoidalPositionEmbeddings(torch.nn.Module):
    def __init__(self, dim, device):
        super().__init__()
        self.dim = dim
        self.device = device

    def forward(self, time):
        res = torch.zeros((len(time), self.dim), dtype=torch.float32, device=self.device)

        pos = torch.arange(len(time)).unsqueeze(1)
        two_i = torch.arange(0, self.dim, 2)

        arg = torch.exp(-two_i * torch.log(torch.tensor(10 ** 4)) / self.dim)

        res[:, 0::2] = torch.sin(pos * arg)
        res[:, 1::2] = torch.cos(pos * arg)

        return res


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = 500
    N = 200
    embed = SinusoidalPositionEmbeddings(d, device)
    res = embed(torch.linspace(0.001, 1, N))

    plt.figure(figsize=(15, 8))
    plt.imshow(res.cpu().numpy())
    plt.xlabel(f"Embedding dimension - d={d}")
    plt.ylabel(f"Original dimension - time position - N={N}")
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.show()

# TODO: add discussion and nice plot of embedding
