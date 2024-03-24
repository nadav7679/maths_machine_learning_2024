import torch
import torch.nn as nn

from time_embed import SinusoidalPositionEmbeddings


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
        x = self.bnorm1(x)  # TODO: unsure about this but ok

        x = x + t[:, :, None, None]
        x = self.relu(self.conv2(x))
        x = self.bnorm2(x)  # TODO: again unsure

        return self.transform(x)


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self):
        global IMG_SIZE

        super().__init__()
        self.depth = 2  # Has to be below 3 to avoid problems with image size HW
        image_channels = 1
        down_channels = [ 2 * (2 ** i) for i in range(self.depth + 1)] # These are the channels that we want to obtain in the downsampling stage; DEFINE YOURSELF!
        up_channels = [ (2 * 2 ** (self.depth-i)) for i in range(self.depth + 1)] # These are the channels that we want to obtain in the upsampling stage; DEFINE YOURSELF!
        out_dim = image_channels # DEFINE THIS CORRECTLY
        time_emb_dim = IMG_SIZE # DEFINE THIS CORRECTLY

        # Time embedding consists of a Sinusoidal embedding, a linear map that maintains the dimensions and a rectified linear unit activation.
        self.time_embed = SinusoidalPositionEmbeddings(time_emb_dim)

        # Initial projection consisting of a map from image_channels to down_channels[0] with a filter size of e.g. 3 and padding of 1.
        self.project_up = nn.Conv2d(image_channels, down_channels[0], (3, 3), padding=(1, 1))

        # Downsample: use the Blocks given above to define down_channels number of downsampling operations. These operations should cha
        # TO WRITE CODE HERE; HINT: use something like Block(down_channels[i], down_channels[i+1], time_emb_dim) the right number of times.
        self.down = []
        for i in range(self.depth):
            self.down.append(Block(down_channels[i], down_channels[i+1], time_emb_dim))

        # Upsample
        self.up = []
        for i in range(self.depth):
            self.up.append(Block(up_channels[i], up_channels[i+1], time_emb_dim, True))

        # Final output: given by a final convolution that maps up_channels[-1] to out_dim with a kernel of size 1.
        self.project_down = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_embed(timestep)
        # Initial conv
        x = self.project_up(x)

        # Unet: iterate through the downsampling operations and the upsampling operations. Do not forget to include the residual connections
        # between the outputs from the downsample stage and the upsample stage.
        residuals = []
        for i in range(self.depth):
            x = self.down[i](x, t)
            residuals.append(x)

        for i in range(self.depth):
            x = torch.concat((x, residuals[-(i+1)]), 1)
            x = self.up[i](x, t)

        return self.project_down(x)


if __name__ == "__main__":
    unet = SimpleUnet()

    batch_size = 5
    x = torch.rand((batch_size, 1, IMG_SIZE, IMG_SIZE))
    t = torch.randperm(batch_size)

    unetted_x = unet(x, t)
    print(unetted_x)