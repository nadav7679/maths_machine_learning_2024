import torch
import matplotlib.pyplot as plt


class Deepdream:
    """
    Deepdream class for generating dream-like images using Inception 3 PyTorch.
    """

    def __init__(self, target_modules, iterations, lr):
        """
        Initialize Deepdream with target modules (layers to optimize), iterations, and learning rate.

        :param target_modules: List of target module names to focus on during optimization.
        :param iterations: Number of iterations for optimization.
        :param lr: Learning rate for optimization.
        """
        self.target_modules = target_modules
        self.iterations = iterations
        self.lr = lr

        # Load pre-trained InceptionV3 model
        self.model: torch.nn.Module = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        self.model.eval()

        # Extract all modules and references from the model
        modules = {}
        for name, module in self.model.named_modules():
            modules[name] = module

        #: A dictionary with (key,value) = (module name, reference (pointer) to module) for all modules.
        self.modules = modules
        #: A dictionary with (key,value) = (module name, module activation output)
        self.activation = {}

        # Register the target layers in the forward pass
        for module_name in target_modules:
            self.modules[module_name].register_forward_hook(self.get_activation(module_name))

    def get_activation(self, module_name):
        """
        Using closure (on self.activation) to save the activations outputs of a certain module.

        :param module_name: The name of the module we are targeting.
        :return: Callable function hook.
        """

        def hook(model, input, output):
            self.activation[module_name] = output

        return hook

    def deepdream(self, x: torch.Tensor, normalize=False):
        """
        Generate deep dream image.

        :param x: Input image tensor.
        :param normalize: Flag to normalize gradient.
        :return: A list of dream image tensor on each iteration.
        """

        x = x.unsqueeze(0) if len(x.shape) == 3 else x
        x.requires_grad = True

        loss = torch.nn.MSELoss(reduction="sum")
        x_seq = [x.clone()]
        for _ in range(self.iterations):
            # Forward pass input data
            self.model(x)

            # Calculate target layers loss
            tot_loss = torch.tensor(0., requires_grad=True)
            for _, a in self.activation.items():
                a_loss = loss(a, torch.zeros_like(a))
                a_loss.retain_grad()
                tot_loss = tot_loss + a_loss

            tot_loss.backward()

            g = x.grad
            if normalize:
                g -= g.mean()
                g /= g.std()
            else:
                g *= torch.abs(g).mean()

            with torch.no_grad():
                x += self.lr * g.detach()
                torch.clip(x, 0, 1, out=x)

            self.model.zero_grad()

            x_seq.append(x.clone())

        return x_seq

    @staticmethod
    def get_module_names():
        """
        Get the list of module names in the pre-trained InceptionV3 model.

        :return: List of module names.
        """
        model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        return [name for name, _ in model.named_modules()]


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
    deepdream = Deepdream(
        [Deepdream.get_module_names()[5], Deepdream.get_module_names()[6]],
        10,
        0.000001
    )

    null_dream = deepdream.deepdream(x=torch.ones([3, 299, 299]))
    show_images(torch.ones([3, 299, 299]), null_dream[0])
