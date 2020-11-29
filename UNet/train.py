import torch
import torch.nn as nn
import torch.utils.data.dataloader as dataloader
from torchvision.transforms import transforms
from unet_model_ import *
from PatchGAN import *

from datasets import ImageDataset

# transform = [
#     # transforms.Resize((args.image_height, args.image_width), interpolation=Image.BICUBIC),
#     # transforms.RandomCrop(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ]
# dataset = ImageDataset(root=, transforms_=transform)
# data = torch.utils.data.DataLoader(
#     dataset,
#     batch_size=,
#     shuffle=,
#     num_workers=,
#
# )
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Generator = Unet_model(in_c=3, out_c=3).to(device)
discriminator_Global = discriminator(in_c=3).to(device)
# define loss type
criterionGAN = nn.MSELoss().cuda()

# define optimizer
optimizer_G = torch.optim.Adam(Generator.parameters(), lr=0.001)
optimizer_D = torch.optim.Adam(discriminator_Global.parameters(), lr=0.001)

epochs = 200

for epoch in range(epochs):
    for index, data in enumerate():
        input_A = data['A'].cuda()
        real_B = data['B'].cuda()
        input_gray = data['A_gray'].cuda()

        fake_b = Generator(input_A, input_gray)

        # update D
        loss_D = 0.0
        Generator.set_requires_grad(discriminator_Global, True)
        optimizer_D.zero_grad()
        pred_fake = discriminator_Global(fake_b.detach())
        pred_real = discriminator_Global(real_B)
        loss_D_Global = criterionGAN((pred_real - torch.mean(pred_fake)), True) + criterionGAN(  # RAGAN
            pred_fake - torch.mean(pred_real), False)
        loss_local = 0
        loss_D = loss_D_Global + loss_local  # + VGG feature map loss
        loss_D.backward()
        optimizer_D.step()

        # update G
        loss_G = 0
        Generator.set_requires_grad(discriminator_Global, False)
        optimizer_G.zero_grad()
        loss_G_Global = criterionGAN(pred_fake - torch.mean(pred_real), True) + criterionGAN(
            pred_real - torch.mean(pred_fake), False)
        loss_G_local = 0
        loss_G = loss_G_Global + loss_G_local  # + VGG feature map loss
        loss_G.backward()
        optimizer_G.step()

        # print loss
        # save checkpoint
