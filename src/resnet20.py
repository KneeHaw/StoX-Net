import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from src.StochasticLibQuan import StoX_Conv2d


def _weights_init(m):
    classname = m.__class__.__name__
    print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock_StoX(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stox_params, stride=1):
        super(BasicBlock_StoX, self).__init__()
        self.conv1 = StoX_Conv2d(in_planes, planes, stox_params, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = StoX_Conv2d(planes, planes, stox_params, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4),
                                                        "constant", 0))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out += self.shortcut(x)
        out = x1 = F.hardtanh(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x1
        out = F.hardtanh(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, stox_params, in_channels, num_classes=10):
        # stox_params = [stox_params_linear, stox_params_conv, model_params]
        super(ResNet, self).__init__()

        self.in_planes = 16
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv1 = StoX_Conv2d(in_channels, 16, stox_params[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stox_params[1], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stox_params[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stox_params[1], stride=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stox_params, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stox_params, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.hardtanh(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.bn2(out)
        out = self.linear(out)
        return out


def resnet20(stox_params_linear, stox_params_conv, model_params, in_c):
    return ResNet(BasicBlock_StoX, [3, 3, 3], [stox_params_linear, stox_params_conv, model_params], in_c)
