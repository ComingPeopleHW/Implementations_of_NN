from LeNet import MNIST_dataload as main
import torch
import torch.nn as nn
import torch.optim as optim
import time
import pandas as pd
import numpy as np

'''
LeNet has 2 convolution and 3 full-connected layers.we can change parameters in __inif__() for different datasets
(like MNIST)classify
'''
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1Layer = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.conv2Layer = nn.Sequential(  # in leCun paper , conv2 layer was not connect every conv1 feature map
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.fcLayer1 = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU())
        self.fcLayer2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU())
        self.fcLayer3 = nn.Sequential(
            nn.Linear(84, 10))

    def forward(self, x):
        x = self.conv1Layer(x)
        x = self.conv2Layer(x)
        x = x.reshape(x.size(0), -1)
        x = self.fcLayer1(x)
        x = self.fcLayer2(x)
        x = self.fcLayer3(x)
        return x


def train(model, device, train_loader, epoch):
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    for epoch in range(epoch):
        r_loss = 0.0
        for i, data in enumerate(train_loader, start=0):
            inputs, lables = data[0].to(device), data[1].to(device)
            # print(i,inputs.size())
            net.zero_grad()
            outputs = model(inputs).to(device)
            loss = criterion(outputs, lables).to(device)
            loss.backward()
            optimizer.step()
            r_loss += loss.item()
            if i % 1000 == 999:
                print('[%d, %5d] loss: %f' % (epoch + 1, i + 1, r_loss / (1000 * 20)))
                r_loss = 0.0
    mnist_path = '../data/MNIST/LeNet.pth'
    torch.save(net.state_dict(), mnist_path)
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


trainloader, testloader = main.MNIST_dataset()  # load MINIST dataset
device = torch.device("cuda:0" if torch.cuda.is_available() else 'CPU')
start_t = time.time()
net = LeNet().to(device)
train_accuracy = train(net, device, trainloader, 5)
test_accuracy = test(net, device, testloader)
end_t = time.time()
cost_t = end_t - start_t
print('train finish,cost time:', str(cost_t))
scores = pd.read_csv(r'../data/MNIST/score.txt', header=None, sep=' ')
if np.array(scores)[0, 1] < float(test_accuracy * 100):
    scores[0], scores[1], scores[2] = round(train_accuracy, 4), test_accuracy, round(cost_t, 4)
    scores.to_csv('./data/score.txt', sep=' ', header=None, index=False)
