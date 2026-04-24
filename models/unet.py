import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        if diff_y or diff_x:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                             diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard UNet with configurable depth.

    Args:
        in_channels:  Number of input channels.
                      - Dual-view (frontal + lateral concatenated): 2
                      - Single-view (one angle at a time): 1
        out_channels: Number of output channels (one per view or 1 for single-view).
        features:     Channel sizes at each encoder depth.
        bilinear:     Use bilinear upsampling; if False uses transposed convolutions.
    """
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        features: tuple = (64, 128, 256, 512),
        bilinear: bool = True,
    ):
        super().__init__()
        f = features
        self.inc   = DoubleConv(in_channels, f[0])
        self.down1 = Down(f[0], f[1])
        self.down2 = Down(f[1], f[2])
        self.down3 = Down(f[2], f[3])
        self.down4 = Down(f[3], f[3] * 2)

        self.up1 = Up(f[3] * 2 + f[3], f[3], bilinear=bilinear)
        self.up2 = Up(f[3] + f[2],     f[2], bilinear=bilinear)
        self.up3 = Up(f[2] + f[1],     f[1], bilinear=bilinear)
        self.up4 = Up(f[1] + f[0],     f[0], bilinear=bilinear)
        self.outc = OutConv(f[0], out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.outc(x)