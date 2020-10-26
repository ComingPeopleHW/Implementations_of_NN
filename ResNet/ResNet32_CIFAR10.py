import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet_32(nn.Module):
    """
    implementation of resnet_34 for CIFAR10 datasets in paper 'Deep Residual Learning for Image Recognition'
    section 4.2,but to sovle the different dimension of the block,we use opiton 'B'(convolution) not the option 'A'(in p
    aper)
    """

    def __init__(self, BasicBlock, num_BasicBlock, num_classes=10):
        super(ResNet_32, self).__init__()
        self.init_in_c = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16))
        self.conv2_x = self.maker_layer(BasicBlock, 16, num_BasicBlock[0], stride=1)
        self.conv3_x = self.maker_layer(BasicBlock, 32, num_BasicBlock[1], stride=2)
        self.conv4_x = self.maker_layer(BasicBlock, 64, num_BasicBlock[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * BasicBlock.expansion, num_classes)

    def maker_layer(self, BasicBlock, out_c, num_BasicBlock, stride):
        strides = [stride] + [1] * (num_BasicBlock - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.init_in_c, out_c, stride))
            self.init_in_c = BasicBlock.expansion * out_c
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_c, out_c, stride=1):
        super(BasicBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c * BasicBlock.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c))

        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c * BasicBlock.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, BasicBlock.expansion * out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(BasicBlock.expansion * out_c))

    def forward(self, x):
        out = self.conv(x)
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out
