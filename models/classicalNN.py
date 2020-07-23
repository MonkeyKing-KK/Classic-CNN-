import torch
import torch.nn as nn
from torchstat import stat


class ConvReLU(nn.Module):
    def __init__(self, inp, oup, kernel_size):
        super(ConvReLU, self).__init__()
        k = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=kernel_size, padding=k),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        layer = [nn.Sequential(
            ConvReLU(3, 64, 3),
            ConvReLU(64, 64, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvReLU(64, 128, 3),
            ConvReLU(128, 128, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvReLU(128, 256, 3),
            ConvReLU(256, 256, 3),
            ConvReLU(256, 256, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvReLU(256, 512, 3),
            ConvReLU(512, 512, 3),
            ConvReLU(512, 512, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvReLU(512, 512, 3),
            ConvReLU(512, 512, 3),
            ConvReLU(512, 512, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )]
        self.features = nn.Sequential(*layer)

        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return self.classifier(out)

