import torch
import torch.nn as nn
import torch.utils.data.dataloader as dataloader
from torchvision.transforms import transforms
from networks import *
from PatchGAN import *
import random
from PIL import Image
from vgg import *
from dataset import ImageDataset
import time

zoom = 1 + 0.1 * random.randint(0, 4)
transform = [
    transforms.Resize((int(zoom * 400), int(zoom * 600)), interpolation=Image.BICUBIC),
    transforms.RandomCrop(256),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
dataset = ImageDataset(root='data', transforms=transform)
data = torch.utils.data.DataLoader(
    dataset,
    batch_size=20,
    shuffle=True,
    num_workers=20
)
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
criterionGAN2 = GANLoss()
vggloss = PerceptualLoss()

# define parameters
epochs = 200
epoch_save = 20
num_patch = 5.0
patch_size = 32
lr = 0.001

# define optimizer
optimizer_G = torch.optim.Adam(Generator.parameters(), lr=lr)
optimizer_D = torch.optim.Adam(d_Global.parameters(), lr=lr)
optimizer_D_local = torch.optim.Adam(d_local.parameters(), lr=lr)
torch.autograd.set_detect_anomaly(True)
for epoch in range(epochs):
    epoch_start_time = time.time()
    for index, data_item in enumerate(data):
        input_A = data_item['A'].cuda()
        real_B = data_item['B'].cuda()
        input_gray = data_item['A_gray'].cuda()
        # print('input_A shape:', input_A.shape, ' real_B:', real_B.shape, 'input_gray:', input_gray.shape)
        # print('input_gray shape:', input_gray.shape)
        fake_B = Generator(input_A, input_gray)
        # print('fake_b:', fake_B.shape)
        real_normal_patch, enhanced_patch, input_patch = get_Patch(int(num_patch), patch_size, real_B, fake_B, input_A)

        # update D
        loss_D = 0.0
        set_requires_grad(d_Global, True)
        optimizer_D.zero_grad()
        optimizer_D_local.zero_grad()
        pred_real = d_Global(real_B)
        pred_fake = d_Global(fake_B.detach())
        loss_D_Global = (criterionGAN2((pred_real - torch.mean(pred_fake)), False) + criterionGAN2(  # RAGAN
            pred_fake - torch.mean(pred_real), True)) / 2.0
        loss_local = 0.0
        for i in range(int(num_patch)):
            pred_real_patch = d_local(real_normal_patch[i])
            pred_fake_patch = d_local(enhanced_patch[i].detach())
            loss_d_real = criterionGAN(pred_real_patch, True)
            loss_d_fake = criterionGAN(pred_fake_patch, False)
            loss_local = loss_local + (loss_d_fake + loss_d_real) / 2.0
        loss_local = loss_local / num_patch
        loss_D = loss_D_Global + loss_local
        loss_D.backward()
        optimizer_D.step()
        optimizer_D_local.step()
        # end update D---------------------------------------------------------
        # test
        # update G
        loss_G = 0
        set_requires_grad(d_Global, False)
        optimizer_G.zero_grad()
        pred_real = d_Global(real_B)
        pred_fake = d_Global(fake_B)
        loss_G_Global = (criterionGAN(pred_real - torch.mean(pred_fake), False) +
                         criterionGAN((pred_fake - torch.mean(pred_real)), True)) / 2.0

        loss_G_local = 0
        for i in range(int(num_patch)):
            pred_fake_patch = d_local(enhanced_patch[i])
            loss_G_local = loss_G_local + criterionGAN(pred_fake_patch, True)
        loss_G_local = loss_G_local / num_patch

        # vgg global & local loss
        vgg_local_loss = 0.0
        vgg_global_loss = vggloss.compute_vgg_loss(vgg, input_A, fake_B)
        for i in range(int(num_patch)):
            vgg_local_loss = vgg_local_loss + vggloss.compute_vgg_loss(vgg, enhanced_patch[i], input_patch[i])
        vgg_local_loss = vgg_local_loss / num_patch
        vgg_loss = vgg_global_loss + vgg_local_loss

        loss_G = loss_G_Global + loss_G_local + vgg_loss  # + VGG feature map loss

        loss_G.backward()
        optimizer_G.step()
        # print("\r" + 'loss_G: {0} , loss_D : {1}'.format(loss_G, loss_D), end="", flush=True)
        print("\rEpoch: [{}/{}] Batch: [{}/{}] G_Loss: {} D_Loss: {}".format(
            epoch,
            epochs,
            index,
            len(data),
            loss_G.item(),
            loss_D.item()), flush=True, end="")

    if (epoch + 1) % epoch_save == 0:
        save_network(Generator, epoch)

    # update learningRate
    if epoch > 100:
        lr = update_learning_rate(lr, optimizer_G, optimizer_D, optimizer_D_local)

    print('End of epoch, time taken : %d sec' % (time.time() - epoch_start_time))

    # print loss
    # save checkpoint
