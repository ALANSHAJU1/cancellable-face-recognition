import torch
import torch.nn as nn
from torchvision.models import resnet18

# -------------------------------
# SE Block
# -------------------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc2(torch.relu(self.fc1(y)))
        y = torch.sigmoid(y).view(b, c, 1, 1)
        return x * y


# -------------------------------
# Extracting Network (EN)
# -------------------------------
class ExtractingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet18(pretrained=True)

        self.features = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )

        self.se = SEBlock(512)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 512)

    def forward(self, x):
        x = self.features(x)
        x = self.se(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.fc(x)
        return x
