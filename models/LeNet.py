import torch
import torch.nn as nn
from torchstat import stat


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            nn.MaxPool2d(2, 2),
        )

        self.layer3 = nn.Sequential(
            nn.Linear(400, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        return self.layer3(out)


# model = LeNet()
# stat(model, (1, 28, 28))
