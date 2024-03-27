import io
import requests

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision import transforms

from deepdream import Deepdream


def load_image(url):
    # TODO: first preprocess, then show image

    img = requests.get(url).content

    with io.BytesIO(img) as file:
        img = mpimg.imread(file, format="jpg")

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
    ])
    img = np.transpose(img, (2, 0, 1))

    return preprocess(torch.tensor(img))


def show_images(img, mod_img, layer=""):

    fig, axes = plt.subplots(1, 2)
    fig.suptitle(layer)

    axes[0].imshow(mod_img.detach().numpy().transpose(1, 2, 0))
    axes[0].set_axis_off()
    axes[0].set_title("Modified")

    axes[1].imshow(img.detach().numpy().transpose(1, 2, 0))
    axes[1].set_axis_off()
    axes[1].set_title("Original")

    plt.show()

mountain = "https://p.turbosquid.com/ts-thumb/fe/3EpLlf/wvg0RHEw/snowmountainprimary01/jpg/1393965432/300x300/sharp_fit_q85/81fd18aacd19ae2e2efdc51dc3de40f5e8f95034/snowmountainprimary01.jpg"
doggie = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
doggie: torch.Tensor = load_image(mountain)
print(doggie.shape)
module_list = Deepdream.get_module_names()
for i, name in enumerate(module_list):
    print(i, name)

target_layers = [module_list[18], module_list[97]]  # Mixed5b, Mixed6b - really nice on mountain!
# target_layers = [module_list[97], module_list[128]]  # Mixed_6b, Mixed 6c
deepdream = Deepdream(target_layers, 200, 1e-03)
mod_img = deepdream.deepdream(doggie.clone(), True)[0]
show_images(doggie, mod_img)
# for i, (name, module) in enumerate(modules[::-1]):
#     try:
#         mod_img = deepdream_optim(4, 0.01, doggie.clone(), module, name, normalize=True)[0]
#         show_images(doggie, mod_img, f"Layer number {i} of type {name}")
#     except:
#         continue
# Cool configs: (l,iter,lr)=(4,5,0.015), (5,5,0.1)