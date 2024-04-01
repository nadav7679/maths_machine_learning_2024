import torch
import torch.nn as nn

from time_embedding import SinusoidalPositionEmbeddings

IMG_SIZE = 28
BATCH_SIZE = 128


class Block(nn.Module):
    """
    A block in the UNet architecture.
    """

    def __init__(self, in_channels, out_channels, time_emb_dim, activation, upsample, up=False, dropout=None):
        """
        Initializes a block in the UNet architecture.

        Args:
            in_channels (int): Input channels to the block.
            out_channels (int): Output channels from the block.
            time_emb_dim (int): The dimensions for the Sinusoidal time embedding.
            activation (torch.nn.Module): Activation function.
            upsample (bool): If it's an up block, whether to use convTranspose or nearest-neighbor upsample.
            up (bool, optional): Up block or down block. Defaults to False.
            dropout (float, optional): Probability of dropout. If None then no dropout is added to the net.
        """
        super().__init__()

        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        if up:
            self.conv1 = nn.Conv2d(2 * in_channels, out_channels, 3, padding=1)

            if upsample:
                self.transform = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                )
            else:
                self.transform = nn.ConvTranspose2d(out_channels, out_channels, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.transform = nn.Conv2d(out_channels, out_channels, 4, 2, 1)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_channels)
        self.bnorm2 = nn.BatchNorm2d(out_channels)

        self.activation = activation

        self.use_dropout = (dropout is not None)
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x, t):
        """
        Defines the forward pass of the block.

        Args:
            x (torch.Tensor): Input tensor.
            t (torch.Tensor): Time tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        t = self.activation(self.time_mlp(t))
        x = self.activation(self.conv1(x))
        x = self.bnorm1(x)

        if self.use_dropout:
            x = self.dropout(x)

        x = x + t[:, :, None, None]  # Broadcasting t to x's shape and summing
        x = self.activation(self.conv2(x))
        x = self.bnorm2(x)

        return self.transform(x)


class SimpleUnet(nn.Module):
    """
    A configurable UNet architecture.
    """

    def __init__(self, base_channels, device, upsample=False, dropout=None, silu=False):
        """
        Initializes the UNet architecture.

        Args:
            base_channels (int): Number of channels at first layer.
            device (torch.device): Device to run the model on.
            upsample (bool, optional): Whether to use upsample nearest-neighbor or ConvTranspose. Defaults to False.
            dropout (float, optional): Dropout rate. Defaults to None.
            silu (bool, optional): Whether to use SiLU activation or ReLU. Defaults to False.
        """
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
        #: Fully connected network for time embedding.
        self.fc_time = nn.Linear(time_emb_dim, time_emb_dim, device=device)

        #: Initial projection to down_channels[0]
        self.project_up = nn.Conv2d(image_channels, down_channels[0], (3, 3), padding=(1, 1))

        down_blocks = []
        for i in range(self.depth):
            down_blocks.append(Block(
                down_channels[i],
                down_channels[i + 1],
                time_emb_dim,
                self.activation,
                upsample,
                dropout=dropout
            ))

        up_blocks = []
        for i in range(self.depth):
            up_blocks.append(Block(
                up_channels[i],
                up_channels[i + 1],
                time_emb_dim,
                self.activation,
                upsample,
                True,
                dropout=dropout
            ))

        self.down = nn.ModuleList(down_blocks)
        self.up = nn.ModuleList(up_blocks)

        #: Final output: a final convolution that maps up_channels[-1] to out_dim with a kernel of size 1.
        self.project_down = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        """
        Perform forward pass through the U-Net model.

        Args:
            x (torch.Tensor): Input tensor.
            timestep (torch.Tensor): Time step tensor before embedding.

        Returns:
            torch.Tensor: Output tensor.
        """
        t = self.activation(self.fc_time(self.time_embed(timestep)))
        x = self.project_up(x)

        residuals = []
        for i in range(self.depth):
            x = self.down[i](x, t)
            residuals.append(x)

        for i in range(self.depth):
            x = torch.cat((x, residuals.pop()), 1)
            x = self.up[i](x, t)

        return self.project_down(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = SimpleUnet(32, device, upsample=True, dropout=0.1, silu=True).to(device=device)

    batch_size = 5
    x = torch.rand((batch_size, 1, IMG_SIZE, IMG_SIZE), device=device)
    t = torch.randperm(batch_size, device=device)

    unetted_x = unet(x, t)
    print(unetted_x.shape)
