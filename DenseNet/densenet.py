import torch
import torch.nn as nn
import torch.nn.functional as F

"""
implement DenseNet-BC in paper 'Densely Connected Convolutional Networks'
"""


class DenseBlock(nn.Module):
    def __init__(self, in_channel, growth_rate):
        super(DenseBlock, self).__init__()
        _1x1_channel = 4 * growth_rate  # for bottleneck 1x1 conv 'we let each 1X1 convolution produce 4k feature-maps.'
        self.bottleneck = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, _1x1_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(_1x1_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(_1x1_channel, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        out = self.bottleneck(x)
        return torch.cat([out, x], 1)


class Transition(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Transition, self).__init__()
        self.transiton = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        return self.transiton(x)


class DenseNet_BC(nn.Module):
    def __init__(self, denseBlock, num_block, growth_rate, theta=0.5, num_class=10):
        """
        DenseNet_BC network
        :param denseBlock: basic block of DenseNet
        :param num_block:  number of block
        :param growth_rate: K in paper
        :param theta: 'theta' for compression
        :param num_class: number of classes of datasets
        """
        super(DenseNet_BC, self).__init__()
        self.growth_rate = growth_rate
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 2 * self.growth_rate, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(2 * self.growth_rate),
            nn.ReLU(inplace=True))

        in_c = 2 * self.growth_rate
        self.features = nn.Sequential()
        for i in range(len(num_block)):  # last denseblock have not Transition,so range() here is (len-1)
            self.features.add_module('dense_block{0}'.format(i),
                                     self._make_dense_layers(denseBlock, in_c, num_block[i]))
            in_c += num_block[i] * self.growth_rate
            if i != (len(num_block) - 1):
                out_c = int(theta * in_c)
                self.features.add_module('Transition{0}'.format(i), Transition(in_c, out_c))
                in_c = out_c
        self.features.add_module('last_normalization', nn.BatchNorm2d(in_c))

        self.fc = nn.Linear(in_c, num_class)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def _make_dense_layers(self, block, in_channels, nblocks):
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module('dense_layer_{}'.format(index), block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return dense_block

    def forward(self, x):
        out = self.conv1(x)
        out = self.features(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def denseNet_BC40():
    return DenseNet_BC(DenseBlock, [12, 12, 12], 12)

def denseNet_BC100():
    return DenseNet_BC(DenseBlock, [32, 32, 32], 12)


def denseNet_BC250():
    return DenseNet_BC(DenseBlock, [82, 82, 82], 24)


def denseNet_BC190():
    return DenseNet_BC(DenseBlock, [62, 62, 62], 40)


net = denseNet_BC40()
print(net)
