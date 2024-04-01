import torch
from torch import nn
from diffusion import Diffusion
import matplotlib.pyplot as plt

from main import train


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
T = 5000
diffusion = Diffusion(T, 30, device, False, False, True, None, True)



# Load model
diffusion.unet = torch.load("diffusion/models/unet_TRANSPOSE_T1000_BC30_E40_ReLU_LIN_P817687_SGD_BS128_DONone.tr")
diffusion.unet.train()

optimizer = torch.optim.SGD(diffusion.unet.parameters(), 0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 4, 0.1)
train(diffusion, 12, optimizer, scheduler, device)
torch.save(diffusion.unet, "unet_TRANSPOSE_T1000_BC30_E40_ReLU_LIN_P817687_ADAMW_SGD_TRY_BS128_DONone.tr")
# #
# for m in diffusion.unet.modules():
#     # print(m)
#     for child in m.children():
#         if type(child) == nn.BatchNorm2d:
#             child.track_running_stats = False
#             child.running_mean = None
#             child.running_var = None
#
# diffusion.unet.eval()

# Sample from model
imgs = diffusion.sample([10, 1, 28, 28])

plot_image_sequence(imgs, 10)

"""
Notes:
Currently, unet_T1000_BC15_E30_cosine_low_dropout_2 and unet_T1000_BC15_E35_cosine works well without 'eval' mode and with T=500, T=1000 respectivly.
unet_T1000_BC32_E35_COS_ADAM_LRSTEP works less well though better then the ones below. Was very hard to train though
Models that don't work well: unet_T1000_BC32_E35_ADAM_LRSTEP, unet_T1000_BC15_E30_cosine_dropout, unet_T1000_BC15_E35_ADAM_LRSTEP, unet_T1000_BC14_E15_cosine_LN.
In fact, unet_T1000_BC14_E15_cosine_LN was shit both in evaluate and train modes.

unet_TRANSPOSE_T1000_BC32_E20_ReLU_LIN_P929409_ADAMW was trained with 128 batch size and is pretty good, although the
loss values are terrible.

unet_TRANSPOSE_T1000_BC32_E20_SiLU_LIN_P929409_ADAMW_BS128 is slightly worse than the ReLU version. Has potential though.

unet_TRANSPOSE_T1000_BC32_E20_ReLU_COS_P929409_ADAMW_BS128 is fucking shit

unet_UPSAMPLE_T1000_BC32_E20_ReLU_LIN_P893569_ADAMW_BS128 gave good loss results but poor images. Seems like it is lacking something,
is it `almost succeeds`.

unet_TRANSPOSE_T1000_BC30_E40_ReLU_LIN_P817687_ADAMW_SGD_TRY_BS128_DONone is surprisingly bad. I guess those extra 
params are important.

I followed the following thread to understand the .eval() problem:
https://discuss.pytorch.org/t/performance-highly-degraded-when-eval-is-activated-in-the-test-phase/3323/67
Eventually adding this code snippet helped, although didn't do a big change like I thought:
for m in diffusion.unet.modules():
    for child in m.children():
        if type(child) == nn.BatchNorm2d:
            child.track_running_stats = False
            child.running_mean = None
            child.running_var = None

diffusion.unet.eval()

Apperantly when you call .eval(), it changes some parameters in batchnorm that we don't want to change. We can also
just not call .eval(), or not save/load the model, as apparently it affects the batchnorm parameters.

Great article on checkerboard effect: https://distill.pub/2016/deconv-checkerboard/
Used upsample similar to here: https://huggingface.co/blog/annotated-diffusion

Seems like dropout shouldn't be too big. Also, seems like cosine works well. The effect of BC15 and BC32 architectures is unclear, although seems like BC15 works better. 
Seems like 'eval' changes drastically the results of the model. It **turns off batchnorm** and dropout layers. It also removes the chess-board phenomena.
It is interesting to see what would happen upon switching the order of batchnorms, it might remove the checkerbox pattern.
"""