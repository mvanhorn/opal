"""ResNet-18 for 32x32 images (3x3 stem, no max-pool; width 64 = 11.2M parameters), as in examples/cifar10_resnet.py
and torchvision's resnet18 with a CIFAR stem. No dropout."""

import torch
import torch.nn as nn
import torch.nn.functional as F

TAU = 0.04


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(out)) + self.shortcut(x))


class ResNet18(nn.Module):
    def __init__(self, width=64, num_classes=10):
        super().__init__()
        w = [width, 2 * width, 4 * width, 8 * width]
        self.stem = nn.Sequential(nn.Conv2d(3, w[0], 3, 1, 1, bias=False), nn.BatchNorm2d(w[0]), nn.ReLU())
        layers, cin = [], w[0]
        for i, cout in enumerate(w):
            layers += [BasicBlock(cin, cout, 1 if i == 0 else 2), BasicBlock(cout, cout, 1)]
            cin = cout
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(w[-1], num_classes)

    def forward(self, x):
        x = self.layers(self.stem(x))
        h = torch.flatten(F.adaptive_avg_pool2d(x, 1), 1).float()
        with torch.autocast(device_type=h.device.type, enabled=False):
            z = self.fc(h)
        z = z - z.mean(dim=1, keepdim=True)
        return z / (TAU * z.norm(dim=1, keepdim=True))
