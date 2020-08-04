import torch
import torch.nn as nn
import math


class fireModule(nn.Module):
    def __init__(self, inp, oup, ratio=4):
        super(fireModule, self).__init__()
        expand_channel = int(oup/2)
        squeeze_channel = int(expand_channel/ratio)
        
        self.squeeze = nn.Conv2d(inp, squeeze_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.expand_1x1 = nn.Conv2d(squeeze_channel, expand_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.expand_3x3 = nn.Conv2d(squeeze_channel, expand_channel, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.squeeze(x))
        out_1x1 = self.expand_1x1(out)
        out_3x3 = self.expand_3x3(out)
        out = self.relu(torch.cat([out_1x1, out_3x3], dim=1))
        return out


class squeezeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(squeezeNet, self).__init__()
        layers = [nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )]

        input_channel = 96
        block = fireModule
        output_channel = [128, 128, 256, 256, 384, 384, 512, 512]
        for i in range(8):
            layers.append(block(inp=input_channel, oup=output_channel[i]))
            input_channel = output_channel[i]
            if i == 2 or i == 6:
                layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        self.features = nn.Sequential(*layers)

        self.conv10 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(output_channel[7], num_classes, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        out = self.features(x)
        out = self.conv10(out)
        out = self.avgpool(out)
        return out
