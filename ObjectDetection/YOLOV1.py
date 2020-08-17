'''naive implementation of YOLOV1'''

import torch
import torch.nn as nn


class Conv1x1LReLU(nn.Module):
    def __init__(self, inp, oup):
        super(Conv1x1LReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Conv3x3LReLU(nn.Module):
    def __init__(self, inp, oup):
        super(Conv3x3LReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class YOLOV1(nn.Module):
    def __init__(self):
        super(YOLOV1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv3x3LReLU(64, 192),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv1x1LReLU(192, 128),
            Conv3x3LReLU(128, 256),
            Conv1x1LReLU(256, 256),
            Conv3x3LReLU(256, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv1x1LReLU(512, 256),
            Conv3x3LReLU(256, 512),
            Conv1x1LReLU(512, 256),
            Conv3x3LReLU(256, 512),
            Conv1x1LReLU(512, 256),
            Conv3x3LReLU(256, 512),
            Conv1x1LReLU(512, 256),
            Conv3x3LReLU(256, 512),
            Conv1x1LReLU(512, 512),
            Conv3x3LReLU(512, 1024),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv1x1LReLU(1024, 512),
            Conv3x3LReLU(512, 1024),
            Conv1x1LReLU(1024, 512),
            Conv3x3LReLU(512, 1024),
            Conv3x3LReLU(1024, 1024),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            Conv3x3LReLU(1024, 1024),
            Conv3x3LReLU(1024, 1024),
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024*7*7, 4096),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 7*7*30),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(x.size(0), -1)
        out = self.classifier(out)
        return out


if __name__ == '__main':
    model = YOLOV1()
    print(model)
    
    img = torch.randn(1, 3, 448, 448)
    output = model(img)
    print(output.shape)
