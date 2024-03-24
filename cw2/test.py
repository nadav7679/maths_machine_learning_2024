import torch
from diffusion import Diffusion
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 500
diffusion = Diffusion(T, device)

# Load model
diffusion.unet = torch.load("./unet_T500_C6_E20.tr")

# Sample from model
imgs = diffusion.sample([10, 1, 28, 28])
img_seq_0 = [imgs_t[0] for imgs_t in imgs]
print(len(img_seq_0))
# for i in reversed(range(100)):
#     plt.imshow(img_seq_0[i][0])
#     plt.pause(0.1)
#     plt.clf()
plt.imshow(img_seq_0[10][0])
plt.show()

