import torch
import torch.nn as nn
from ConvolutionBlock import ConvolutionBlock
from IdentityBlock import IdentityBlock

class ResNet_18(nn.Module):
    def __init__(self):
        super(ResNet_18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2, padding = 3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )
        self.conv2 = nn.Sequential(
            ConvolutionBlock(in_channels = 64, out_channels = 64),
            IdentityBlock(in_channels = 64, out_channels = 64)
        )
        self.conv3 = nn.Sequential(
            ConvolutionBlock(in_channels = 64, out_channels = 128),
            IdentityBlock(in_channels = 128, out_channels = 128)
        )
        self.conv4 = nn.Sequential(
            ConvolutionBlock(in_channels = 128, out_channels = 256),
            IdentityBlock(in_channels = 256, out_channels = 256)
        )
        self.conv5 = nn.Sequential(
            ConvolutionBlock(in_channels = 256, out_channels = 512),
            IdentityBlock(in_channels = 512, out_channels = 512)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        return out