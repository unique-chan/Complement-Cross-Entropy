import torch
import torch.nn as nn
import torch.nn.functional as F

from I_configuration import NUM_CLASSES


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        identity = x                                # [input]
        out = self.conv1(x)                         # -> [3x3, 64]
        out = self.bn1(out)                         # -> (bn)
        out = F.relu(out)                           # -> [relu]
        out = self.conv2(out)                       # -> [3x3, 64]
        out = self.bn2(out)                         # -> (bn)
        out = out + self.shortcut(identity)         # -> + [input]
        out = F.relu(out)                           # -> [relu]
        return out                                  # :[out]


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * self.expansion,
                               kernel_size=1, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels * self.expansion)
            )

    def forward(self, x):
        identity = x                                # [input]
        out = self.conv1(x)                         # -> [1x1, 64]
        out = self.bn1(out)                         # -> (bn)
        out = F.relu(out)                           # -> [relu]
        out = self.conv2(out)                       # -> [3x3, 64]
        out = self.bn2(out)                         # -> (bn)
        out = F.relu(out)                           # -> [out]
        out = self.conv3(out)                       # -> [1x1, 256]
        out = self.bn3(out)                         # -> (bn)
        out = out + self.shortcut(identity)         # -> + [input]
        out = F.relu(out)                           # -> [relu]
        return out                                  # :[out]


class ResNet(nn.Module):
    def _create_layer(self, block, out_channels, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        sub_layers = []
        for stride in strides:
            sub_layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*sub_layers)

    def __init__(self, block, num_blocks, num_classes=NUM_CLASSES):
        super(ResNet, self).__init__()

        self.in_channels = 64
        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_features=64)

        self.layer1 = self._create_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._create_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._create_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._create_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(in_features=512 * block.expansion, out_features=num_classes)

    def forward(self, x):
        # conv1
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        # conv2_x
        out = self.layer1(out)
        # conv3_x
        out = self.layer2(out)
        # conv4_x
        out = self.layer3(out)
        # conv5_x
        out = self.layer4(out)
        # 1x1 avg pool
        # out = F.avg_pool2d(input=out, kernel_size=4)
        out = nn.AdaptiveAvgPool2d((1, 1))(out)
        out = out.view(out.size()[0], -1)
        out = self.linear(out)
        # For softmax activation in training, modify given III_train.py.
        # (As you know, valid/test do not need the softmax activation.)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet101():
    return ResNet(BottleNeck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(BottleNeck, [3, 8, 36, 3])


def test():
    model = ResNet18()
    y = model(torch.randn(1, 3, 32, 32))
    print(y.size())


# test()
