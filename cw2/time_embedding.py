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

        arg = torch.exp(torch.log(pos) - two_i * torch.log(torch.tensor(10 ** 4)) / self.dim)

        res[:, 0::2] = torch.sin(arg)
        res[:, 1::2] = torch.cos(arg)

        return res


if __name__ == "__main__":
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
