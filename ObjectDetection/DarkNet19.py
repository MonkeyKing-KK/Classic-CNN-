import torch
import torch.nn as nn


class Conv1x1BNLReLU(nn.Module):
    def __init__(self, inp, oup):
        super(Conv1x1BNLReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(oup),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Conv3x3BNLReLU(nn.Module):
    def __init__(self, inp, oup):
        super(Conv3x3BNLReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(oup),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Darknet19(nn.Module):
    def __init__(self, num_classes=1000):
        super(Darknet19, self).__init__()
        self.stage1 = nn.Sequential(
            Conv3x3BNLReLU(3, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv3x3BNLReLU(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.stage2 = nn.Sequential(
            Conv3x3BNLReLU(64, 128),
            Conv1x1BNLReLU(128, 64),
            Conv3x3BNLReLU(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv3x3BNLReLU(128, 256),
            Conv1x1BNLReLU(256, 128),
            Conv3x3BNLReLU(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.stage3 = nn.Sequential(
            Conv3x3BNLReLU(256, 512),
            Conv1x1BNLReLU(512, 256),
            Conv3x3BNLReLU(256, 512),
            Conv1x1BNLReLU(512, 256),
            Conv3x3BNLReLU(256, 512),
        )

        self.stage4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv3x3BNLReLU(512, 1024),
            Conv1x1BNLReLU(1024, 512),
            Conv3x3BNLReLU(512, 1024),
            Conv1x1BNLReLU(1024, 512),
            Conv3x3BNLReLU(512, 1024),
            # Conv3x3BNLReLU(1024, 1024),
            # Conv3x3BNLReLU(1024, 1024),
        )

        self.stage5 = nn.Sequential(
            Conv3x3BNLReLU(1024, num_classes),
            nn.AvgPool2d(kernel_size=7, stride=1),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.stage1(x)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.stage5(out)
        out = torch.squeeze(out, dim=3).contiguous()
        out = torch.squeeze(out, dim=2).contiguous()
        out = self.softmax(out)
        return out


if __name__ == '__main__':
    model = Darknet19()
    print(model)
    
    img = torch.randn(1, 3, 224, 224)
    out = model(img)
    print(out.shape)
