import torch
import torch.nn as nn
import torch.utils.data.dataloader as dataloader
from torchvision.transforms import transforms
from networks import *
from PatchGAN import *
import random
from PIL import Image
from UNet.vgg import *
from datasets import ImageDataset

zoom = 1 + 0.1 * random.randint(0, 4)
transform = [
    transforms.Resize((int(zoom * 400), int(zoom * 600)), interpolation=Image.BICUBIC),
    transforms.RandomCrop(256),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
# dataset = ImageDataset(root=, transforms_=transform)
# data = torch.utils.data.DataLoader(
#     dataset,
#     batch_size=,
#     shuffle=,
#     num_workers=,
#
# )
# define networks
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# generator
Generator = Unet_model(in_c=3, out_c=3).to(device)
# discriminator local & global
d_Global = discriminator(in_c=3).to(device)
d_Global.apply(weights_init)
d_local = discriminator(in_c=3).to(device)
d_local.apply(weights_init)

vgg = load_vgg('./models')
# define loss type
criterionGAN = GANLoss()
vggloss = PerceptualLoss()

# define optimizer
optimizer_G = torch.optim.Adam(Generator.parameters(), lr=0.001)
optimizer_D = torch.optim.Adam(d_Global.parameters(), lr=0.001)
optimizer_D_local = torch.optim.Adam(d_local.parameters(), lr=0.001)

epochs = 200
epoch_save = 20
num_patch = 5.0
patch_size = 32

for epoch in range(epochs):
    for index, data in enumerate():
        input_A = data['A'].cuda()
        real_B = data['B'].cuda()
        input_gray = data['A_gray'].cuda()

        fake_b = Generator(input_A, input_gray)
        real_normal_patch, enhanced_patch, input_patch = get_Patch(num_patch, patch_size, real_B, fake_b, input_A)

        # update D
        loss_D = 0.0
        set_requires_grad(d_Global, True)
        optimizer_D.zero_grad()
        pred_fake = d_Global(fake_b.detach())
        pred_real = d_Global(real_B)
        loss_D_Global = (criterionGAN((pred_real - torch.mean(pred_fake)), False) + criterionGAN(  # RAGAN
            pred_fake - torch.mean(pred_real), True)) / 2.0

        loss_local = 0.0
        for i in range(int(num_patch)):
            pred_real = d_local(real_normal_patch[i])
            pred_fake = d_local(enhanced_patch[i].detach())
            loss_d_real = criterionGAN(pred_real, True)
            loss_d_fake = criterionGAN(pred_fake, False)
            loss_local += (loss_d_fake + loss_d_real) / 2.0
        loss_local /= num_patch
        loss_D = loss_D_Global + loss_local
        loss_D.backward()
        optimizer_D.step()
        optimizer_D_local.step()

        # update G
        loss_G = 0
        set_requires_grad(d_Global, False)
        optimizer_G.zero_grad()
        loss_G_Global = 0.0
        loss_G_Global = (criterionGAN(pred_fake - torch.mean(pred_real), True) + criterionGAN(
            pred_real - torch.mean(pred_fake), False)) / 2
        loss_G_local = 0
        for i in range(int(num_patch)):
            pred_fake_patch = d_local(enhanced_patch[i])
            loss_G_local += criterionGAN(pred_fake_patch, True)
        loss_G_local /= num_patch

        # vgg global & local loss
        vgg_local_loss = 0.0
        vgg_global_loss = vggloss(vgg, input_A, fake_b)
        for i in range(int(num_patch)):
            vgg_local_loss += vggloss(enhanced_patch[i], input_patch[i])
        vgg_local_loss /= num_patch
        vgg_loss = vgg_global_loss + vgg_local_loss

        loss_G = loss_G_Global + loss_G_local + vgg_loss  # + VGG feature map loss
        loss_G.backward()
        optimizer_G.step()

    if (epoch + 1) % epoch_save == 0:
        Generator.save_network(Generator, epoch)

        # print loss
        # save checkpoint
