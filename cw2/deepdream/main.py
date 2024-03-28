import io
import requests

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision import transforms

from deepdream import Deepdream


def load_image(url):
    """
    Load image from URL and preprocess it.

    :param url: URL of the image.
    :return: Preprocessed image tensor.
    """

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


def plot_iteration_grid(imgs, stepsize, img_names=None, title=""):
    """
    Display a batch of images sequences.

    :param imgs: A list of tensors. Essentialy an nxm matrix, each row is m iterations of the same image. Each entry
     in imgs is a tensor of size [batch_size, 3, H, W].
    :param stepsize: Stepsize on iterations.
    :param img_names: List of length batch_size with names for the sequences.
    :param title: title (optional).
    """
    row_length = (len(imgs) // stepsize) + 1
    fig, axes = plt.subplots(imgs[0].shape[0], row_length, figsize=(15, 8))
    fig.suptitle(title, y=0.9)

    if img_names is None:
        img_names = ["" for _ in range(imgs[0].shape[0])]

    if imgs[0].shape[0] == 1:
        axes = [axes]

    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            ax.imshow(imgs[j][i].detach().numpy().transpose(1, 2, 0))
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

            if not i:
                ax.set_title(f"iteration {j * stepsize}")

            if not j:
                ax.set_ylabel(f"{img_names[i]}")

    fig.tight_layout()
    plt.show()


# Get images
mountain_url = "https://p.turbosquid.com/ts-thumb/fe/3EpLlf/wvg0RHEw/snowmountainprimary01/jpg/1393965432/300x300/sharp_fit_q85/81fd18aacd19ae2e2efdc51dc3de40f5e8f95034/snowmountainprimary01.jpg"
doggie_url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
pug_url = "https://media.istockphoto.com/id/185094249/photo/little-fat-pug-sitting-on-sidewalk-in-summer-park.jpg?s=612x612&w=0&k=20&c=vfvypCBLX7OrPFEM58QLGBd9cuhTEN0SET55z3P31Wo="
pizza_url = "https://thumbs.dreamstime.com/b/high-resolution-pizza-image-white-background-sliced-pepperoni-pizza-toppings-captured-timeless-artistry-style-287201104.jpg"

doggie: torch.Tensor = load_image(doggie_url).unsqueeze(0)
pug: torch.Tensor = load_image(pug_url).unsqueeze(0)
pizza: torch.Tensor = load_image(pizza_url).unsqueeze(0)
mountain: torch.Tensor = load_image(mountain_url).unsqueeze(0)

# Get module names
module_list = Deepdream.get_module_names()

# Choose target layers
comb1 = [module_list[50], module_list[65], module_list[31], module_list[84]]
comb2 = [module_list[137], module_list[158], module_list[115]]
comb3 = [module_list[241], module_list[286], module_list[295]]

# Create Deepdream instances
deepdream1 = Deepdream(comb1, 20, 5e-02)
deepdream2 = Deepdream(comb2, 20, 5e-02)
deepdream3 = Deepdream(comb3, 20, 5e-02)

# Generate and display modified images
dream1 = deepdream1.deepdream(torch.cat((mountain.clone(), doggie.clone()), 0), True)
dream2 = deepdream2.deepdream(mountain.clone(), True)
dream3 = deepdream3.deepdream(doggie.clone(), True)

# Plots
plot_iteration_grid(dream1, 5, ["mountain", "doggie"], title="18, 7")
plot_iteration_grid(dream2, 5, ["mountain"], title="Layers: 50, 65, 31, 84")
plot_iteration_grid(dream3, 5, ["doggie"], title="283, 250, 220")