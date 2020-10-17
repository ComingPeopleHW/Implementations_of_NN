import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from Alex import Fashion_MNIST_loader as dload
import pandas as pd


# the original parameters of AlexNet in ALex paper are different form follow,because we
# want to apply this model on Fashion_MNIST whici is a gray image dataset.
# AlexNet(in=3,out=96,k=11,s=4)
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1Layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75),  # the parameters in paper
            nn.MaxPool2d(2, 2))  # AlexNet(k=3,s=2)
        self.conv2Layer = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # AlexNet(k=5,s=1,p=2)
            nn.ReLU(),
            # nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75),
            nn.MaxPool2d(2, 2))  # AlexNet(k=3,s=2)
        self.conv3Layer = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.conv4Layer = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.conv5Layer = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2))  # AlexNet(k=3,s=2)
        self.fc1Layer = nn.Sequential(  # 3 full-connected layers,and softmax in last
            nn.Linear(256 * 3 * 3, 1024),  # AlexNet(6*6*256,4096)
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc2Layer = nn.Sequential(
            nn.Linear(1024, 512),  # AlexNet(4096,4096)
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc3Layer = nn.Sequential(
            nn.Linear(512, 10))  # AlexNet(4096,1000)
        # nn.Softmax(dim=1))

    def forward(self, x):
        x = self.conv1Layer(x)
        x = self.conv2Layer(x)
        x = self.conv3Layer(x)
        x = self.conv4Layer(x)
        x = self.conv5Layer(x)
        x = x.reshape(x.size(0), -1)
        # x = x.view(-1, 256 * 3 * 3)
        x = self.fc1Layer(x)
        x = self.fc2Layer(x)
        x = self.fc3Layer(x)
        return x


def train(model, device, train_loader, epoch):
    criterion = nn.CrossEntropyLoss().to(device)
    # optimizer = optim.Adam(net.parameters(), lr=0.01)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    for epoch in range(epoch):
        r_loss = 0.0
        for i, data in enumerate(train_loader, start=0):
            inputs, lables = data[0].to(device), data[1].to(device)
            # print(inputs.size())
            net.zero_grad()
            outputs = model(inputs).to(device)
            loss = criterion(outputs, lables).to(device)
            loss.backward()
            optimizer.step()
            r_loss += loss.item()
            if i % 1000 == 999:
                print('[%d, %5d] loss: %f' % (epoch + 1, i + 1, r_loss / (1000 * 20)))
                r_loss = 0.0
    mnist_path = '../data/FashionMNIST/Fashion_MNISTbyAlexNet.pth'
    torch.save(model.state_dict(), mnist_path)
    print('trained model saved done')
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(train_loader):
            inputs, lables = data[0].to(device), data[1].to(device)
            outputs = model(inputs).to(device)
            _, pred = torch.max(outputs, dim=1)
            correct += (pred == lables.view_as(pred)).sum().item()
    train_accuracy = float(correct) / len(train_loader.dataset)
    print('test set accuracy: ', str(train_accuracy))
    return train_accuracy


def test(model, device, test_loader):
    correct = 0
    for i, data in enumerate(test_loader):
        inputs, lables = data[0].to(device), data[1].to(device)
        outputs = model(inputs).to(device)
        _, pred = torch.max(outputs, dim=1)
        correct += (pred == lables.view_as(pred)).sum().item()
    test_accuracy = float(correct) / len(test_loader.dataset)
    print('train set accuracy: ', str(test_accuracy))
    return test_accuracy


trainloader, testloader = dload.Fashion_MNIST_dataset()  # load MINIST dataset
device = torch.device("cuda:0" if torch.cuda.is_available() else 'CPU')
start_t = time.time()
net = AlexNet().to(device)
train_accuracy = train(net, device, trainloader, epoch=20)
test_accuracy = test(net, device, testloader)
end_t = time.time()
cost_t = end_t - start_t
print('train finish,cost time:', str(cost_t))
scores = pd.read_csv(r'../data/FashionMNIST/score.txt', header=None, sep=' ')
if np.array(scores)[0, 1] < float(test_accuracy):
    scores[0], scores[1], scores[2] = round(train_accuracy, 4), test_accuracy, round(cost_t, 4)
    scores.to_csv('../data/FashionMNIST/score.txt', sep=' ', header=None, index=False)
