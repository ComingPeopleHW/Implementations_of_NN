import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import os


# basic conv in UNet
class double_conv(nn.Module):
    def __init__(self, in_c, out_c, mid_c=None):
        super(double_conv, self).__init__()
        mid_c = out_c
        self.double_convolution = nn.Sequential(
            nn.Conv2d(in_c, mid_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.double_convolution(x)


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
        # self.maxpool_conv = nn.Sequential(
        #     nn.MaxPool2d(2),
        #     double_conv(in_channels, out_channels)
        # )
        self.maxpool = nn.MaxPool2d(2)

        self.conv = double_conv(in_channels, out_channels)

    def forward(self, x, gray=None):
        x = self.maxpool(x)
        if gray is not None:
            # print(gray.shape)
            # print(x.shape)
            x = x * gray
        x = self.conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1)
        )
        self.conv = double_conv(in_channels, out_channels)  # because torch.cat, so here inchannels is 'in_channels'

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x


class Unet_model(nn.Module):
    def __init__(self, in_c, out_c):
        super(Unet_model, self).__init__()
        self.max_pool2d = nn.MaxPool2d(2)
        self.ini_conv = double_conv(in_c, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)

        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 32)
        self.out = OutConv(32, out_c)

    def forward(self, input_, gray):
        # gray_x for attention map
        gray_2 = self.max_pool2d(gray)
        gray_3 = self.max_pool2d(gray_2)
        gray_4 = self.max_pool2d(gray_3)
        gray_5 = self.max_pool2d(gray_4)
        x1 = self.ini_conv(input_)  # gray
        x2 = self.down1(x1)  # gray1
        x3 = self.down2(x2)  # gray2
        x4 = self.down3(x3)  # gray3
        x5 = self.down4(x4, gray_5)  # gray4

        x4 = x4 * gray_4
        x = self.up1(x5, x4)

        x3 = x3 * gray_3
        x = self.up2(x, x3)

        x2 = x2 * gray_2
        x = self.up3(x, x2)

        x1 = x1 * gray
        x = self.up4(x, x1)

        latent = self.out(x)
        latent = latent * gray
        output = latent + input_
        return output


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, tensor=torch.cuda.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input_, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input_.numel()))
            if create_label:
                real_tensor = self.Tensor(input_.size()).fill_(self.real_label)
                # self.real_label_var = torch.tensor(real_tensor, requires_grad=True)
                self.real_label_var = real_tensor.clone().detach().requires_grad_(True)
                # self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input_.numel()))
            if create_label:
                # print('false,create')
                fake_tensor = self.Tensor(input_.size()).fill_(self.fake_label)
                # self.fake_label_var = torch.tensor(fake_tensor, requires_grad=True)
                self.fake_label_var = fake_tensor.clone().detach().requires_grad_(True)

                # self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input_, target_is_real):
        target_tensor = self.get_target_tensor(input_, target_is_real)
        return self.loss(input_, target_tensor)


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def save_network(net_G, epoch):
    save_filename = '%d_.pth' % epoch
    save_dir = 'checkpoints'
    save_path = os.path.join(save_dir, save_filename)
    torch.save(net_G, save_path)


def get_Patch(num_patch, patch_size, realB_data, fakeB_data, input_data):
    w = realB_data.size(3)
    h = realB_data.size(2)
    real_patch = []
    fake_patch = []
    input_patch = []
    for i in range(num_patch):
        w_offset = random.randint(0, max(0, w - patch_size - 1))
        h_offset = random.randint(0, max(0, h - patch_size - 1))
        real_patch.append(realB_data[:, :, h_offset:h_offset + patch_size, w_offset:w_offset + patch_size])
        fake_patch.append(fakeB_data[:, :, h_offset:h_offset + patch_size, w_offset:w_offset + patch_size])
        input_patch.append(input_data[:, :, h_offset:h_offset + patch_size, w_offset:w_offset + patch_size])
    return real_patch, fake_patch, input_patch


def update_learning_rate(lr, G, D, D_local):
    old_lr = lr
    lrd = lr / 100  # opt.lr default=0.0001
    lr = lr - lrd
    for param_group in D.param_groups:
        param_group['lr'] = lr

    for param_group in D_local.param_groups:
        param_group['lr'] = lr
    for param_group in G.param_groups:
        param_group['lr'] = lr

    print('update learning rate: %f -> %f' % (old_lr, lr))
    return lr
