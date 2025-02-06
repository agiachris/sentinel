import torch.nn as nn
import torch.nn.functional as F


class DecoderResNet(nn.Module):

    def __init__(
        self, in_dim=3, out_dim=1, h_dim=128, num_layers=5, leaky=False, **kwargs
    ):
        super().__init__()

        self.fc_in = nn.Linear(in_dim, h_dim)

        self.num_layers = num_layers
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            self.blocks.append(ResnetBlockFC(h_dim))

        self.fc_out = nn.Linear(h_dim, out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, x):
        """Performs a forward pass through the network.

        Args:
            p (tensor): points tensor
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        """
        net = self.fc_in(x)

        for i in range(self.num_layers):
            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))

        return out


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    """Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx
