import torch
import torch.nn as nn

from time_embedding import SinusoidalPositionEmbeddings

IMG_SIZE = 28
BATCH_SIZE = 128


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        """
        in_ch refers to the number of channels in the input to the operation and out_ch how many should be in the output
        """
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)

        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        """
        Define the forward pass making use of the components above.
        Time t should get mapped through the time_mlp layer + a relu
        The input x should get mapped through a convolutional layer with relu / batchnorm
        The time embedding should get added the output from the input convolution
        A second convolution should be applied and finally passed through the self.transform.
        """
        t = self.relu(self.time_mlp(t))
        x = self.relu(self.conv1(x))
        x = self.bnorm1(x)

        x = x + t[:, :, None, None]
        x = self.relu(self.conv2(x))
        x = self.bnorm2(x)

        return self.transform(x)


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """

    def __init__(self, base_channels, device):
        super().__init__()

        #: The 'depth' of the unet, i.e. the amount of down blocks and up blocks.
        #: It has to be below 3 to avoid various problems with image size H,W (e.g. H,W shrinks to 1,1 which is too
        #: small to kernel size, or unmatching H,W between same-level up and down blocks)
        self.depth = 2
        #: Device
        self.device = device

        image_channels = 1
        down_channels = [base_channels * (2 ** i) for i in range(self.depth + 1)]
        up_channels = [(base_channels * 2 ** (self.depth - i)) for i in range(self.depth + 1)]
        out_dim = image_channels
        time_emb_dim = 32

        #: Sinusoidal time embedding
        self.time_embed = SinusoidalPositionEmbeddings(time_emb_dim, device)

        #: Initial projection to down_channels[0]
        self.project_up = nn.Conv2d(image_channels, down_channels[0], (3, 3), padding=(1, 1))

        # Downsample and upsample
        down = []
        for i in range(self.depth):
            down.append(Block(down_channels[i], down_channels[i + 1], time_emb_dim))

        up = []
        for i in range(self.depth):
            up.append(Block(up_channels[i], up_channels[i + 1], time_emb_dim, True))

        self.down = nn.ModuleList(down)
        self.up = nn.ModuleList(up)

        #: Final output: a final convolution that maps up_channels[-1] to out_dim with a kernel of size 1.
        self.project_down = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_embed(timestep)
        # Initial conv
        x = self.project_up(x)

        residuals = []
        for i in range(self.depth):
            x = self.down[i](x, t)
            residuals.append(x)

        for i in range(self.depth):
            x = torch.concat((x, residuals.pop()), 1)
            x = self.up[i](x, t)

        return self.project_down(x)


if __name__ == "__main__":
    unet = SimpleUnet()

    batch_size = 5
    x = torch.rand((batch_size, 1, IMG_SIZE, IMG_SIZE))
    t = torch.randperm(batch_size)

    unetted_x = unet(x, t)
    print(unetted_x)
