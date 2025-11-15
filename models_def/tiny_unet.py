import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv → BN → ReLU) twice"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    """Tiny UNet for lane segmentation"""
    def __init__(self):
        super().__init__()

        # Encoder
        self.conv1 = DoubleConv(3, 16)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.conv3 = DoubleConv(32, 64)

        # Decoder
        self.up4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv4 = DoubleConv(64, 32)

        self.up5 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.conv5 = DoubleConv(32, 16)

        # Output: 1 mask channel
        self.out = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        # Bottleneck
        c3 = self.conv3(p2)

        # Decoder
        u4 = self.up4(c3)
        u4 = torch.cat([u4, c2], dim=1)
        c4 = self.conv4(u4)

        u5 = self.up5(c4)
        u5 = torch.cat([u5, c1], dim=1)
        c5 = self.conv5(u5)

        return torch.sigmoid(self.out(c5))
