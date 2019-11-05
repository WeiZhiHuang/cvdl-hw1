import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

import os
import copy


EPOCHS = 1
BATCH_SIZE = 64
PRINT_FREQ = 100
TRAIN_NUMS = 49000

CUDA = True

PATH_TO_SAVE_DATA = './'
ABS_PATH = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(ABS_PATH, 'model.pkl')


data_transform = transforms.Compose([transforms.ToTensor()])

train_data = datasets.CIFAR10(root=PATH_TO_SAVE_DATA, train=True,
                              download=True, transform=data_transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                          sampler=SubsetRandomSampler(range(TRAIN_NUMS)))
val_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                        sampler=SubsetRandomSampler(range(TRAIN_NUMS, 50000)))


if CUDA:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')
print(device)


class Trainer:
    def __init__(self, criterion, optimizer, device):
        self.criterion = criterion
        self.optimizer = optimizer

        self.device = device

    def train_loop(self, model, train_loader, val_loader):
        results = []
        for epoch in range(EPOCHS):
            print('---------------- Epoch {} ----------------'.format(epoch))
            self._training_step(model, train_loader, epoch)
            results.append(self._validate(model, val_loader, epoch))
        return results

    def _training_step(self, model, loader, epoch):
        model.train()

        for step, (X, y) in enumerate(loader):
            X, y = X.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            outs = model(X)
            loss = self.criterion(outs, y)

            if step >= 0 and (step % PRINT_FREQ == 0):
                self._state_logging(outs, y, loss, step, epoch, 'Training')

            loss.backward()
            self.optimizer.step()

    def _validate(self, model, loader, epoch, state='Validate'):
        model.eval()
        outs_list = []
        loss_list = []
        y_list = []

        with torch.no_grad():
            for step, (X, y) in enumerate(loader):
                X, y = X.to(self.device), y.to(self.device)

                outs = model(X)
                loss = self.criterion(outs, y)

                y_list.append(y)
                outs_list.append(outs)
                loss_list.append(loss)

            y = torch.cat(y_list)
            outs = torch.cat(outs_list)
            loss = torch.mean(torch.stack(loss_list), dim=0)
            self._state_logging(outs, y, loss, step, epoch, state)

        return (self._accuracy(outs, y), loss)

    def _state_logging(self, outs, y, loss, step, epoch, state):
        acc = self._accuracy(outs, y)
        print('[{:3d}/{}] {} Step {:03d} Loss {:.3f} Acc {:.3f}'.format(epoch +
                                                                        1, EPOCHS, state, step, loss, acc))

    def _accuracy(self, output, target):
        batch_size = target.size(0)

        pred = output.argmax(1)
        correct = pred.eq(target)
        acc = correct.float().sum(0) / batch_size

        return acc

    def train_one_epoch(self, model, train_loader):
        losses = []
        model.train()
        for _, (X, y) in enumerate(train_loader):
            X, y = X.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            outs = model(X)
            loss = self.criterion(outs, y)
            losses.append(loss)
            loss.backward()
            self.optimizer.step()
        return losses


def getTrainImages():
    imgs = []
    for img, label in train_loader:
        imgs.append((transforms.ToPILImage()(img[0]), label[0]))
        if len(imgs) == 10:
            break
    return imgs


def startTrainOneEpoch():
    return trainer.train_one_epoch(copy.deepcopy(clearModel), train_loader)


def loadModel():
    if os.path.isfile(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        return True
    return False


def startTrainLoop():
    if not loadModel():
        result = trainer.train_loop(model, train_loader, val_loader)
        torch.save(model.state_dict(), MODEL_PATH)

        fig = plt.figure(1)
        ax1 = plt.subplot(211)
        ax1.set(title='Accurancy', ylabel='%')
        plt.plot(range(EPOCHS), np.array(result)[:, 0])
        ax2 = plt.subplot(212)
        ax2.set(xlabel='epoch', ylabel='loss')
        plt.plot(range(EPOCHS), np.array(result)[:, 1])
        fig.savefig(os.path.join(ABS_PATH, 'result.png'))
        plt.clf()


def inference(imgIndex):
    if loadModel():
        plt.figure(figsize=(12, 4))
        plt.gcf().canvas.set_window_title('')
        plt.subplot(121)
        plt.imshow(train_loader.dataset.data[imgIndex])
        img = data_transform(train_loader.dataset.data[imgIndex]).float()
        img = torch.tensor(img, requires_grad=True).unsqueeze(0)
        predict = model(img).detach()
        return nn.Softmax(dim=1)(predict).numpy()[0]
    return np.array([])


model = nn.Sequential(nn.Conv2d(3, 6, 5),
                      nn.ReLU(),
                      nn.MaxPool2d(2),
                      nn.Conv2d(6, 16, 5),
                      nn.ReLU(),
                      nn.MaxPool2d(2),
                      nn.Flatten(),
                      nn.Linear(400, 120),
                      nn.ReLU(),
                      nn.Linear(120, 84),
                      nn.ReLU(),
                      nn.Linear(84, 10))

clearModel = copy.deepcopy(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    params=model.parameters(), lr=1e-3, momentum=0.9)
trainer = Trainer(criterion, optimizer, device)
