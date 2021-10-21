import torch
import torchvision
from torch import nn

import numpy as np


####Implement a simple UNet
## composed of contract blocks and expanding blocks
## Reference: https://medium.com/analytics-vidhya/creating-a-very-simple-u-net-model-with-pytorch-for-semantic-segmentation-of-satellite-images-223aa216e705


class UNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = self.downscale_block(in_channels, 32, 7, 3)
        self.conv2 = self.downscale_block(32, 64)
        self.conv3 = self.downscale_block(64, 128)

        self.up_conv3 = self.upscale_block(128, 64)
        self.up_conv2 = self.upscale_block(64 * 2, 32)
        self.up_conv1 = self.upscale_block(32 * 2, out_channels)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)

        up3 = self.up_conv3(c3)
        up2 = self.up_conv2(torch.cat([up3, c2], 1))
        up1 = self.up_conv1(torch.cat([up2, c1], 1))

        return up1

    def downscale_block(self, in_channels, out_channels, kernelSize=3, pad=1):
        downscale = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernelSize, stride=1, padding=pad),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernelSize, stride=1, padding=pad),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        return downscale

    def upscale_block(self, in_channels, out_channels):
        upscale = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        return upscale


model = UNet(4, 2)
print(model)

# Check if the output shape is right
# x = torch.ones((64,4,256,256))
# pred = model(x)
# print(pred.shape)
