import torch
import torch.nn as nn
import torch.nn.functional as F

class IdentityBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(IdentityBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1)

    def forward(self, x):
        # output of convolutions
        out = F.relu(self.batchnorm1(self.conv1(x)))
        out = F.relu(self.conv2(out))

        # residual connection
        res = F.relu(x)

        # residual and output
        out = F.relu(out + res)
        return out