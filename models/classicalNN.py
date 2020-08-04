import torch
import torch.nn as nn
import math
from torchstat import stat


class ConvBNReLU(nn.Module):
    def __init__(self, inp, oup, kernel_size):
        super(ConvBNReLU, self).__init__()
        k = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=kernel_size, padding=k),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.clamp(y, 0, 1)
        return x * y


class MyConv(nn.Module):
    def __init__(self, inp, oup, stride=1, ratio=2, split_rate=2):
        super(MyConv, self).__init__()
        self.inp_3x3 = int(inp/split_rate)
        self.inp_1x1 = inp - self.inp_3x3
        self.init_channel = math.ceil(oup/ratio)
        self.new_channel = self.init_channel*(ratio-1)
        self.oup = oup

        self.primary_conv = nn.Sequential(
            nn.Conv2d(self.inp_3x3, self.init_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(self.init_channel),
            nn.ReLU(inplace=True),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(self.init_channel, self.new_channel, kernel_size=3, stride=1, padding=1, groups=self.init_channel, bias=False),
            nn.BatchNorm2d(self.new_channel),
            nn.ReLU(inplace=True),
        )
        self.pwise_low = nn.Conv2d(self.inp_1x1, oup, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        out_low = self.pwise_low(x[:, self.inp_3x3:, :, :])
        x1 = self.primary_conv(x[:, :self.inp_3x3, :, :])
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        out = out_low + out[:, :self.oup, :, :]
        return out


class SPConv_3x3(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, ratio=0.5):
        super(SPConv_3x3, self).__init__()
        self.inplanes_3x3 = int(inplanes*ratio)
        self.inplanes_1x1 = inplanes - self.inplanes_3x3
        self.outplanes_3x3 = int(outplanes*ratio)
        self.outplanes_1x1 = outplanes - self.outplanes_3x3
        self.outplanes = outplanes
        self.stride = stride

        self.gwc = nn.Conv2d(self.inplanes_3x3, self.outplanes, kernel_size=3, stride=self.stride,
                              padding=1, groups=2, bias=False)
        # self.gwc = nn.Conv2d(self.inplanes_3x3, self.outplanes, kernel_size=3, stride=self.stride,
        #                    padding=1, groups=self.inplanes_3x3, bias=False)
        self.pwc = nn.Conv2d(self.inplanes_3x3, self.outplanes, kernel_size=1, bias=False)

        self.conv1x1 = nn.Conv2d(self.inplanes_1x1, self.outplanes, kernel_size=1)
        self.avgpool_s2_1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avgpool_s2_3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avgpool_add_1 = nn.AdaptiveAvgPool2d(1)
        self.avgpool_add_3 = nn.AdaptiveAvgPool2d(1)
        self.bn1 = nn.BatchNorm2d(self.outplanes)
        self.bn2 = nn.BatchNorm2d(self.outplanes)
        self.ratio = ratio
        self.groups = int(1/self.ratio)

    def forward(self, x):
        b, c, _, _ = x.size()
        x_3x3 = x[:, :int(c*self.ratio), :, :]
        x_1x1 = x[:, int(c*self.ratio):, :, :]
        out_3x3_gwc = self.gwc(x_3x3)
        if self.stride == 2:
            x_3x3 = self.avgpool_s2_3(x_3x3)
        out_3x3_pwc = self.pwc(x_3x3)
        out_3x3 = out_3x3_gwc + out_3x3_pwc
        out_3x3 = self.bn1(out_3x3)
        out_3x3_ratio = self.avgpool_add_3(out_3x3).squeeze(dim=3).squeeze(dim=2)

        # use avgpool first to reduce information lost
        if self.stride == 2:
            x_1x1 = self.avgpool_s2_1(x_1x1)

        out_1x1 = self.conv1x1(x_1x1)
        out_1x1 = self.bn2(out_1x1)
        out_1x1_ratio = self.avgpool_add_1(out_1x1).squeeze(dim=3).squeeze(dim=2)

        out_31_ratio = torch.stack((out_3x3_ratio, out_1x1_ratio), 2)
        out_31_ratio = nn.Softmax(dim=2)(out_31_ratio)
        out = out_1x1 * (out_31_ratio[:, :, 1].view(b, self.outplanes, 1, 1).expand_as(out_1x1))\
              + out_3x3 * (out_31_ratio[:, :, 0].view(b, self.outplanes, 1, 1).expand_as(out_3x3))

        return out


class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            ConvBNReLU(3, 64, 3),
            SPConv_3x3(64, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SPConv_3x3(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SPConv_3x3(128, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SPConv_3x3(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SPConv_3x3(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SPConv_3x3(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SPConv_3x3(256, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            SPConv_3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            SPConv_3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            SPConv_3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            SPConv_3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            SPConv_3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
            # nn.Linear(512*7*7, 4096),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            # nn.Linear(4096, 4096),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            # nn.Linear(4096, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return self.classifier(out)


# model = VGG16(num_classes=10)
# stat(model, (3, 32, 32))


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


model = squeezeNet()
stat(model, (3, 224, 224))


