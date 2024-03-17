import io
import requests

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from torchvision import transforms
from PIL import Image


model: torch.nn.Module = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
model.eval()


def load_image(url, show=False):
    # TODO: first preprocess, then show image

    img = requests.get(url).content

    with io.BytesIO(img) as file:
        img = mpimg.imread(file, format="jpg")

    if show:
        plt.imshow(img)
        plt.show()

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
    ])
    img = np.transpose(img, (2, 0, 1))
    return preprocess(torch.tensor(img))


activation = {}
def get_activation(name):
    global activation
    def hook(model, input, output):
        activation[name] = output
    return hook


def deepdream_optim(iterations: int, lr: float, x: torch.Tensor, module, module_name, normalize=False):
    # TODO: make getting the activations less ugly
    global model

    x = x.unsqueeze(0) if len(x.shape)==3 else x
    x.requires_grad = True
    module.register_forward_hook(get_activation(module_name))
    loss = torch.nn.MSELoss(reduction="sum")
    for _ in range(iterations):
        outputs = model(x)
        activations_l = activation[module_name]

        l = loss(activations_l, torch.zeros_like(activations_l))
        # l = torch.norm(torch.flatten(activations_l), 2)
        l.retain_grad()
        l.backward()

        g = x.grad
        if normalize:
            g -= g.mean()
            g /= g.std()

        else:
            g *= torch.abs(g).mean()

        # print(lr * g.detach())
        with torch.no_grad():
            x += lr * g.detach()

        model.zero_grad()
        # x.zero_grad()

    return x

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


if __name__ == "__main__":
    doggie: torch.Tensor = load_image("https://github.com/pytorch/hub/raw/master/images/dog.jpg")

    modules = []
    for name, module in model.named_modules():
        modules.append((name, module))

    print(len(modules))

    # l = 33
    # print(f"Activation layer name: {modules[l]}")
    # mod_img = deepdream_optim(4, 0.1, doggie.clone(), modules[l][1], modules[l][1], True)[0]
    # show_images(doggie, mod_img, f"Layer number {l} of type {modules[l][1]}")
    for i, (name, module) in enumerate(modules[::-1]):
        try:
            mod_img = deepdream_optim(4, 0.01, doggie.clone(), module, name, normalize=True)[0]
            show_images(doggie, mod_img, f"Layer number {i} of type {name}")
        except:
            continue
    # Cool configs: (l,iter,lr)=(4,5,0.015), (5,5,0.1)