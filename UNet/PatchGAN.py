import torch as torch
from torch import nn
import numpy as np
from torch.nn import functional as F


# patch gan discriminator in cycleGAN
class discriminator(nn.Module):
    def __init__(self, in_c):
        # patchGAN的构建
        super(discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_c, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv2d(256, 512, 4, 1, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv2d(512, 1, 4, 1, 1)  # output 1 channel prediction map
        )

    def forward(self, x):
        out = self.model(x)
        # x = F.adaptive_avg_pool2d(x, 1).view(x.size()[0], -1)  # output size (batch_size,1)
        return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
