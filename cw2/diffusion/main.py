import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Compose

from diffusion import Diffusion

BATCH_SIZE = 64

# load dataset from the hub
dataset = load_dataset("fashion_mnist")
channels = 1

# define image transformations (e.g. using torchvision)
transform = Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
])


def transforms(examples):
    examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
    del examples["image"]
    return examples

transformed_dataset = dataset.with_transform(transforms).remove_columns("label")

dataloader = DataLoader(transformed_dataset['train'], batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# TODO: experiment with different betas and hyperparameters for noise addition.


# Main training loop
def train(diffusion: Diffusion, nr_epochs, optimizer, scheduler, device, dataloader=dataloader):
    diffusion.unet.train()
    for epoch in range(nr_epochs):
        # iterate through batches
        for i, data in enumerate(dataloader, 0):

            # get inputs
            images = data["pixel_values"]
            images = images.to(device)
            batch_size = images.shape[0]

            optimizer.zero_grad()

            # Get loss (using forward diffusion and run through unet)
            t = torch.randint(0, diffusion.T, (batch_size, ), device=device, dtype=torch.long)
            loss = diffusion.get_loss(images, t)

            loss.backward()
            optimizer.step()



        # print results for last batch
        print(f"Epoch: {epoch + 1:03} | Loss: {loss:04} | lr: {optimizer.param_groups[0]['lr']}")
        # Update learning_rate
        scheduler.step()

    print('Finished Training')


def plot_iteration_grid(imgs, stepsize, img_names=None, title=""):
    """
    Display a batch of images sequences.

    :param imgs: A list of tensors. Essentialy an nxm matrix, each row is m iterations of the same image. Each entry
     in imgs is a tensor of size [batch_size, 3, H, W].
    :param stepsize: Stepsize on iterations.
    :param img_names: List of length batch_size with names for the sequences.
    :param title: title (optional).
    """
    row_length = (len(imgs) // stepsize)
    fig, axes = plt.subplots(imgs[0].shape[0], row_length, figsize=(15, 8))
    fig.suptitle(title, y=0.9)

    if img_names is None:
        img_names = ["" for _ in range(imgs[0].shape[0])]

    if imgs[0].shape[0] == 1:
        axes = [axes]

    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            ax.imshow(imgs[j][i][0].cpu().detach().numpy())
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

            if not i:
                ax.set_title(f"iteration {j * stepsize}")

            if not j:
                ax.set_ylabel(f"{img_names[i]}")

    fig.tight_layout()
    plt.show()


# Simulate forward diffusion
# torch.manual_seed(2530622)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# T = 20
# diffusion = Diffusion(T, 28, device, True)
#
# batch = next(iter(dataloader))["pixel_values"].to(device=device)
# imgs = batch[4:7]
# num_images = 10
# stepsize = int(T/num_images)
#
# img_seq = [imgs]
# for t in range(0, T):
#     timesteps = torch.ones(imgs.shape[0], device=device, dtype=torch.int64) * t
#     imgs, noise = diffusion.forward_diffusion_sample(imgs, timesteps)
#     img_seq.append(imgs)
#
# plot_iteration_grid(img_seq, stepsize)


# Run training loop
torch.manual_seed(2530622)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
diffusion = Diffusion(1000, 15, device, cosine=False)
# diffusion.unet = torch.load("unet_T1000_BC25_E10.tr")

print(f"Number of parameters: {sum(p.numel() for p in diffusion.unet.parameters() if p.requires_grad)}")

optimizer = torch.optim.Adam(diffusion.unet.parameters(), 0.005)
# optimizer = torch.optim.SGD(diffusion.unet.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, threshold=10E-03)

train(diffusion, 15, optimizer, scheduler, device)
torch.save(diffusion.unet, "diffusion/unet_T1000_BC15_E15_LINTIME_UPSAMPLE_ADAM_LRSTEP.tr")
