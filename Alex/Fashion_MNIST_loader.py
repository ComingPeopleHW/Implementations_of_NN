import numpy as np
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd


# download MNIST dataset,if there has MNIST dataset yet,it will not download again.
def Fashion_MNIST_dataset():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    trainset = torchvision.datasets.FashionMNIST(root='../data', train=True, download=True,
                                                 transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=20, shuffle=True, num_workers=0)
    testset = torchvision.datasets.FashionMNIST(root='../data', train=False, download=True,
                                                transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=20, shuffle=False, num_workers=0)
    return trainloader, testloader


train, test = Fashion_MNIST_dataset()

