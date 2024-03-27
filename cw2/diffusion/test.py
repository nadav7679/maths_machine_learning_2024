import torch
from diffusion import Diffusion
import matplotlib.pyplot as plt


def plot_image_sequence(images, seq_len=10):
    """ Plots some samples from the dataset """
    step_size = len(images) // seq_len
    batch_size = images[0].shape[0]

    print(seq_len, batch_size)
    figure, axes = plt.subplots(batch_size, seq_len + 1)

    for i in range(seq_len + 1):
        t = i * step_size if i != seq_len else i*step_size-1

        batch = images[t]
        axes[0, i].set_title(f"t={i * step_size}")

        for j in range(batch_size):
            axes[j, i].imshow(batch[j][0], cmap="gray")
            axes[j, i].axis(False)


    plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 1000
diffusion = Diffusion(T, 28, device)

# Load model
diffusion.unet = torch.load("diffusion/unet_T1000_C32_E10.tr")
# Currently unet_T1000_C32_E10 works quite well

# Sample from model
imgs = diffusion.sample([10, 1, 28, 28])

plot_image_sequence(imgs, 10)

