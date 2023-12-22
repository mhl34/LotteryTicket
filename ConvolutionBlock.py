import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = 2, padding = 1)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, stride = 2)

    def forward(self, x):
        # residual connection
        res = self.conv3(x)

        # convolutions
        out = self.conv1(x)
        out = F.relu(self.batchnorm1(out))
        out = self.conv2(out)
        out = self.batchnorm2(out)

        # connect the output of convolutions to the residual connection
        out = F.relu(res + out)

        return out
