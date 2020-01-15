#coding=utf-8
import os
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from PIL import Image
import numpy as np
import cv2
import settings
from models import *

class MyDataset(Dataset):
    def __init__(self, folder1, folder2, transform=None):
        self.train_image_file_paths = [os.path.join(folder1, image_file) for image_file in os.listdir(folder1)]
        self.train_label_file_paths = [os.path.join(folder2, txt_file) for txt_file in os.listdir(folder2)]
        self.train_image_file_paths.sort()
        self.train_label_file_paths.sort()
        self.transform = transforms.Compose([
            transforms.Resize((416, 416)),
            transforms.ToTensor()])
    def __len__(self):
        return len(self.train_image_file_paths)
    def __getitem__(self, idx):
        image_root = self.train_image_file_paths[idx]
        image_name = image_root.split(os.path.sep)[-1]
        label_root = self.train_label_file_paths[idx]
        label_name = label_root.split(os.path.sep)[-1]
        image = Image.open(image_root)
        label_path = open(label_root)
        if self.transform is not None:
            image = self.transform(image)
        label = []
        for line in label_path.readlines():
            print(line)
            curLine = line.strip().split("  ")
            x = curLine[0] + curLine[2]
            y = curLine[1] + curLine[3]
            label.append((float)(x))
            label.append((float)(y))
        l = len(label)
        if l < 9:
            for i in range(l, 10):
                label.append(0)
                label.append(0)
        label = torch.Tensor(label)
        return image, label

transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor()])

def get_train_data_loader():
    dataset = MyDataset(settings.TRAIN_DATASET_PATH, settings.TRAIN_LABEL_PATH,transform=transform)
    return DataLoader(dataset, batch_size=2)

def get_test_data_loader():
    dataset = MyDataset(settings.TEST_DATASET_PATH, settings.TEST_LABEL_PATH, transform=transform)
    return DataLoader(dataset, batch_size=2)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 0.001
start_epoch = 0 # start from epoch 0 or last checkpoint epoch
resume = False


trainloader = get_train_data_loader()
testloader = get_test_data_loader()

net = PointNet()
print(net)

if resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']

mse = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs, targets
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = mse(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        print('batch_idx: {}, Loss: {}'.format(batch_idx, train_loss/(batch_idx+1)))

def test(epoch):
    net.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = mse(outputs, targets)
            loss.backward()
            optimizer.step()

            test_loss += loss.item()
            print('batch_idx: {}, Loss: {}'.format(batch_idx, test_loss/(batch_idx+1)))

    state = {
        'net': net.state_dict(),
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.pth')

for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)