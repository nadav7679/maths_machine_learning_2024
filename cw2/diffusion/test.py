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
diffusion = Diffusion(T, 32, device)

# Load model
diffusion.unet = torch.load("diffusion/unet_T1000_BC32_E35_COS_ADAM_LRSTEP.tr")
# diffusion.unet.eval()
# Currently unet_T1000_C32_E10 works quite well

# Sample from model
imgs = diffusion.sample([10, 1, 28, 28])

plot_image_sequence(imgs, 10)

"""
Notes:
Currently, unet_T1000_BC15_E30_cosine_low_dropout_2 and unet_T1000_BC15_E35_cosine works well without 'eval' mode and with T=500, T=1000 respectivly.
unet_T1000_BC32_E35_COS_ADAM_LRSTEP works less well though better then the ones below. Was very hard to train though
Models that don't work well: unet_T1000_BC32_E35_ADAM_LRSTEP, unet_T1000_BC15_E30_cosine_dropout, unet_T1000_BC15_E35_ADAM_LRSTEP.


Seems like dropout shouldn't be too big. Also, seems like cosine works well. The effect of BC15 and BC32 architectures is unclear, although seems like BC15 works better. 
Seems like 'eval' changes drastically the results of the model. It **turns off batchnorm** and dropout layers. It also removes the chess-board phenomena.
It is interesting to see what would happen upon switching the order of batchnorms, it might remove the checkerbox pattern.
"""