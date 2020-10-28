from comet_ml import Experiment
import torch
import torch.nn as nn
import time
from ResNet.ResNet32_CIFAR10 import BasicBlock, ResNet_32
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader


def get_training_dataloader(batch_size=128, num_workers=16, shuffle=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize
    ])
    cifar10_training = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    cifar10_training_loader = DataLoader(cifar10_training, shuffle=shuffle, num_workers=num_workers,
                                         batch_size=batch_size)

    return cifar10_training_loader


def get_test_dataloader(batch_size=128, num_workers=16, shuffle=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])
    cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    cifar10_test_loader = DataLoader(cifar10_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar10_test_loader


def train(_net, _epoch, _train_loader, _optimizer, _criterion, _step):
    _net.train()
    _correct = 0.0
    for i, (_input, label) in enumerate(_train_loader):
        _input = _input.cuda()
        label = label.cuda()

        output = net(_input)
        loss = criterion(output, label)

        _optimizer.zero_grad()
        loss.backward()
        _optimizer.step()

        _, preds = output.max(1)
        _correct += preds.eq(label).sum().float()
        _step += 1
    print("epoch:{0} , train_epoch_acc:{1}".format(_epoch, format(_correct / len(_train_loader.dataset), '.4f')))
    # experiment.log_metric("train_epoch_acc", value=_correct / len(_train_loader.dataset), step=_step)

    return _step


def eval_training(_net, _epoch, test_loader, _step):
    _net.eval()

    correct = 0.0

    for i, (_input, label) in enumerate(test_loader):
        _input = _input.cuda()
        label = label.cuda()
        outputs = _net(_input)

        _, preds = outputs.max(1)
        correct += preds.eq(label).sum().float()
    acc = correct / len(test_loader.dataset)
    print("epoch:{0} , test_epoch_acc:{1}".format(_epoch, format(correct / len(test_loader.dataset), '.4f')))
    experiment.log_metric("test_epoch_acc", value=correct / len(test_loader.dataset), step=_step)
    return acc


if __name__ == '__main__':
    experiment = Experiment(api_key='fxT3Krof2iAW4QWentgNxptou', project_name='ResNet_32_CIFAR10')
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'CPU')
    net = ResNet_32(BasicBlock, [7, 7, 7], num_classes=10).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    epoch = 185
    batch_size = 128
    num_workers = 0

    train_loader = get_training_dataloader(batch_size, num_workers)
    test_loader = get_test_dataloader(batch_size, num_workers)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [95, 140], gamma=0.1)
    start_time = time.time()
    step = 0
    best_acc = 0.0
    for i in range(1, epoch + 1):
        step = train(net, i, train_loader, optimizer, criterion, step)
        t_acc = eval_training(net, i, test_loader, step)
        if t_acc > best_acc:
            best_acc = t_acc
        scheduler.step()
        experiment.log_metric('lr', value=optimizer.param_groups[0]['lr'], step=step)

    end_time = time.time()
    print("time cost: %.4f min" % (float(end_time - start_time) / 60))
    print("best test_set acc : %.4f" % best_acc)
