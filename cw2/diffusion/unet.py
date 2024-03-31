import torch
import torch.nn as nn

from time_embedding import SinusoidalPositionEmbeddings

IMG_SIZE = 28
BATCH_SIZE = 128


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, activation, upsample, up=False, dropout=None):
        """
        in_ch refers to the number of channels in the input to the operation and out_ch how many should be in the output
        """
        super().__init__()

        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)

            if upsample:
                self.transform = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                )
            else:
                self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)

        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)

        self.activation = activation

        self.is_dropout = (dropout is not None)
        if self.is_dropout:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x, t):
        """
        Define the forward pass making use of the components above.
        Time t should get mapped through the time_mlp layer + a relu
        The input x should get mapped through a convolutional layer with relu / batchnorm
        The time embedding should get added the output from the input convolution
        A second convolution should be applied and finally passed through the self.transform.
        """
        t = self.activation(self.time_mlp(t))
        x = self.activation(self.conv1(x))
        x = self.bnorm1(x)

        if self.is_dropout:
            x = self.dropout(x)

        x = x + t[:, :, None, None]
        x = self.activation(self.conv2(x))
        x = self.bnorm2(x)

        return self.transform(x)


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """

    def __init__(self, base_channels, device, upsample=False, dropout=None, silu=False):
        super().__init__()

        #: Device
        self.device = device
        #: The 'depth' of the unet, i.e. the amount of down blocks and up blocks.
        #: It has to be below 3 to avoid various problems with image size H,W (e.g. H,W shrinks to 1,1 which is too
        #: small to kernel size, or unmatching H,W between same-level up and down blocks)
        self.depth = 2

        image_channels = 1
        time_emb_dim = 32
        out_dim = image_channels
        down_channels = [base_channels * (2 ** i) for i in range(self.depth + 1)]
        up_channels = [(base_channels * 2 ** (self.depth - i)) for i in range(self.depth + 1)]

        self.activation = nn.SiLU() if silu else nn.ReLU()

        #: Sinusoidal time embedding
        self.time_embed = SinusoidalPositionEmbeddings(time_emb_dim, device)
        self.fc_time = nn.Linear(time_emb_dim, time_emb_dim, device)

        #: Initial projection to down_channels[0]
        self.project_up = nn.Conv2d(image_channels, down_channels[0], (3, 3), padding=(1, 1))

        # Downsample and upsample
        down = []
        for i in range(self.depth):
            down.append(Block(
                down_channels[i],
                down_channels[i + 1],
                time_emb_dim,
                self.activation,
                upsample,
                dropout=dropout
            ))

        up = []
        for i in range(self.depth):
            # out_pad = 1 if i == 0 else 0
            up.append(Block(
                up_channels[i],
                up_channels[i + 1],
                time_emb_dim,
                self.activation,
                upsample,
                True,
                dropout=dropout
            ))

        self.down = nn.ModuleList(down)
        self.up = nn.ModuleList(up)

        #: Final output: a final convolution that maps up_channels[-1] to out_dim with a kernel of size 1.
        self.project_down = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        t = self.activation(self.fc_time(self.time_embed(timestep)))
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = SimpleUnet(32, device, 0.1, True).to(device=device)

    batch_size = 5
    x = torch.rand((batch_size, 1, IMG_SIZE, IMG_SIZE), device=device)
    t = torch.randperm(batch_size, device=device)

    unetted_x = unet(x, t, )
    print(unetted_x.shape)
