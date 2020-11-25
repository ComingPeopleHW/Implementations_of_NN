import torch
import torch.nn as nn
import torch.nn.functional as F


# conv-block
def double_conv(in_c, out_c, mid_c=None):
    if not mid_c:
        mid_c = out_c
    return nn.Sequential(
        nn.Conv2d(in_c, mid_c, 3, 1, 1),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(mid_c, out_c, 3, 1, 1),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(0.2, inplace=True),
    )


# UNet last layer,output desired channel image
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# downsaple: maxpool wiht double_conv
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1)
        )
        self.conv = double_conv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x


class Unet_model(nn.Module):
    def __init__(self, int_c, out_c):
        super(Unet_model, self).__init__()
        self.ini_conv = double_conv(int_c, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)

        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 32)
        self.out = OutConv(32, out_c)

    def forward(self, x):
        x1 = self.ini_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        return x


net = Unet_model(3, 3)
print(net)
