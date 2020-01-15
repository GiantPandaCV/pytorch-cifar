'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.BN1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.BN2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.BN3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.BN4 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 26 * 26, 128)
        self.Drop1 = nn.Dropout(0.5)
        self.fc2   = nn.Linear(128, 64)
        self.Drop2 = nn.Dropout(0.5)
        self.fc3   = nn.Linear(64, 18)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.BN1(out)
        out = F.max_pool2d(out, 2)

        out = F.relu(self.conv2(out))
        out = self.BN2(out)
        out = F.max_pool2d(out, 2)

        out = F.relu(self.conv3(out))
        out = self.BN3(out)
        out = F.max_pool2d(out, 2)

        out = F.relu(self.conv4(out))
        out = self.BN4(out)
        out = F.max_pool2d(out, 2)

        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.Drop1(out)
        out = F.relu(self.fc2(out))
        out = self.Drop2(out)
        out = self.fc3(out)
        return out
