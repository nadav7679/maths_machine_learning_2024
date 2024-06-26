from argparse import ArgumentParser

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Compose

from diffusion import Diffusion

BATCH_SIZE = 128

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


# Main training loop
def train(diffusion: Diffusion, nr_epochs, optimizer, scheduler, device, dataloader=dataloader, train_samples=False, seed=None, fname=None):
    """
    Train a given diffusion model.

    Args:
        diffusion (Diffusion): Diffusion model.
        nr_epochs (int): Number of epochs.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Scheduler.
        device (torch.device): Device.
        dataloader (torch.utils.data.DataLoader, optional): DataLoader. Defaults to global dataloader.
        train_samples (boolean, optional): If True, return a list with samples from each epoch
        seed (int): seed to sample from
        fname (str, optional): File name for logging. Defaults to None (print to stdout).
    """
    diffusion.unet.train()

    if train_samples:
        epoch_samples = []

    for epoch in range(nr_epochs):
        # Iterate through batches
        for i, data in enumerate(dataloader, 0):
            # Get inputs
            images = data["pixel_values"]
            images = images.to(device)
            batch_size = images.shape[0]

            optimizer.zero_grad()

            # Get loss (using forward diffusion and run through UNet)
            t = torch.randint(0, diffusion.T, (batch_size,), device=device, dtype=torch.long)
            loss = diffusion.get_loss(images, t)

            loss.backward()
            optimizer.step()

        # Get training samples
        if train_samples:
            epoch_samples.append(diffusion.sample([5, 1, 28, 28], seed))

        # Print results for last batch
        if fname is not None:
            with open(f"{fname}.txt", "a") as f:
                print(f"Epoch: {epoch + 1:03} | Loss: {loss:.4f} | lr: {optimizer.param_groups[0]['lr']}", file=f)
        else:
            print(f"Epoch: {epoch + 1:03} | Loss: {loss:.4f} | lr: {optimizer.param_groups[0]['lr']}")

        # Update learning rate
        scheduler.step()

    print("Finished Training")
    if train_samples:
        return train_samples


if __name__ == "__main__":
    parser = ArgumentParser(
        description="""Train a UNet diffusion model.""")
    parser.add_argument("--cosine", action="store_true",
                        help="Use cosine noise")
    parser.add_argument("--silu", action="store_true",
                        help="Use silu instead of relu.")
    parser.add_argument("--upsample", action="store_true",
                        help="Replace transposedconv to upsample with nearest neighbor.")
    parser.add_argument("base_channels", type=int, nargs=1,
                        help="The number of channels in each base level.")
    parser.add_argument("epochs", type=int, nargs=1,
                        help="Number of epochs.")
    parser.add_argument("dropout", type=float, nargs=1,
                        help="chance of dropout.")
    parser.add_argument("lr", type=float, nargs=1,
                        help="Initial lr.")

    args = parser.parse_args()
    base_channels = args.base_channels[0]
    silu = args.silu
    cosine = args.cosine
    upsample = args.upsample
    epochs = args.epochs[0]
    dropout = args.dropout[0] if args.dropout[0] != 0 else None
    lr = args.lr[0]

    torch.manual_seed(2530622)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    diffusion = Diffusion(1000, base_channels, device, upsample, cosine, True, dropout, silu)
    optimizer = torch.optim.AdamW(diffusion.unet.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 25], gamma=0.1)

    param_num = sum(p.numel() for p in diffusion.unet.parameters() if p.requires_grad)
    print(f"Number of parameters: {param_num}")

    fname = (f"diffusion/models/unet_{'UPSAMPLE' if upsample else 'TRANSPOSE'}_T1000_BC{base_channels}_E{epochs}"
             f"_{'Si' if silu else 'Re'}LU_{'COS' if cosine else 'LIN'}_P{param_num}_ADAMW_BS{BATCH_SIZE}_DO{dropout}")
    train(diffusion, epochs, optimizer, scheduler, device, fname=fname)
    torch.save(diffusion.unet, f"{fname}.tr")
