import torch as torch
from torch import nn
import numpy as np
from torch.nn import functional as F

# patch gan discriminator in cycleGAN
class discriminator(nn.Module):
    def __init__(self):
        # patchGAN的构建
        super(discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 512, 4, 1, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 1, 4, 1, 1)
        )
        # 输出通道为1的map

    def forward(self, x):
        x = self.model(x)
        x = F.adaptive_avg_pool2d(x, 1).view(x.size()[0], -1)  # output size (batch_size,1)
        return x


