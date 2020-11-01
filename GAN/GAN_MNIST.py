from comet_ml import Experiment
import torch.nn as nn
import torch
from LeNet.MNIST_dataload import MNIST_dataset
from torch.autograd.variable import Variable
from GAN.utils import Logger

"""
from
https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f
"""


class Discriminator(nn.Module):
    def __init__(self, n_features, n_output):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(n_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3))

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3))

        self.fc3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3))

        self.output = nn.Sequential(
            nn.Linear(256, n_output),
            nn.Sigmoid())

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        out = self.output(x)
        return out


class Generator(nn.Module):
    def __init__(self, n_features, n_output):
        super(Generator, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
        )

        self.output = nn.Sequential(
            nn.Linear(1024, n_output),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        output = self.output(x)
        return output


def images_to_vectors(images):
    return images.view(images.size(0), 784)


def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)


def real_data_target(size):
    return torch.ones(size, 1).cuda()


def fake_data_target(size):
    return torch.zeros(size, 1).cuda()


def train_discriminator(optimizer, real_data, fake_data):
    optimizer.zero_grad()
    pred_real = d_net(real_data)
    real_loss = loss(pred_real, real_data_target(real_data.size(0)))
    real_loss.backward()

    pred_fake = d_net(fake_data)
    fake_loss = loss(pred_fake, fake_data_target(fake_data.size(0)))
    fake_loss.backward()

    optimizer.step()
    return real_loss + fake_loss, pred_real, pred_fake


def train_generator(optimizer, fake_data):
    optimizer.zero_grad()
    pred_fake = d_net(fake_data)
    fake_loss = loss(pred_fake, real_data_target(fake_data.size(0)))
    fake_loss.backward()
    optimizer.step()
    return fake_loss


def noise(size):
    n = torch.randn(size, 100)
    return n.cuda()


if __name__ == '__main__':
    # experiment = Experiment(api_key='fxT3Krof2iAW4QWentgNxptou', project_name='GAN_MNIST')
    g_net = Generator(100, 784).cuda()
    d_net = Discriminator(784, 1).cuda()
    g_optimizer = torch.optim.Adam(g_net.parameters(), lr=0.0002)
    d_optimizer = torch.optim.Adam(d_net.parameters(), lr=0.0002)
    loss = nn.BCELoss().cuda()
    logger = Logger(model_name='VGAN', data_name='MNIST')
    num_test_samples = 16
    test_noise = noise(num_test_samples)
    epochs = 200
    train_loader, _ = MNIST_dataset()
    num_batches = len(train_loader)
    for epoch in range(epochs):
        for i, (real_data, _) in enumerate(train_loader):
            real_data = Variable(images_to_vectors(real_data))
            real_data = real_data.cuda()
            fake_data = g_net(noise(real_data.size(0))).detach()
            d_loss, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, fake_data)
            # experiment.log_metric('d_loss', value=d_loss, step=i)
            # experiment.log_metric('d_pred_real', value=d_pred_fake, step=i)
            # experiment.log_metric('d_pred_fake', value=d_pred_fake, step=i)

            fake_data = g_net(noise(real_data.size(0)))
            g_loss = train_generator(g_optimizer, fake_data)
            # experiment.log_metric('g_loss', value=g_loss, step=i)
            if i % 100 == 0:
                test_images = vectors_to_images(g_net(noise(num_test_samples))).data.cpu()
                logger.log_images(test_images, num_test_samples, epoch, i, num_batches)
        print(epoch)
