import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from src.StochasticLibQuan import StoX_Linear, StoX_Conv2d


def _weights_init(m):
    classname = m.__class__.__name__
    print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LinearNetwork(nn.Module):
    def __init__(self, stox_params_conv, stox_params_linear, model_params):
        super(LinearNetwork, self).__init__()
        self.conv1 = self.conv2 = self.conv3 = self.linear1 = self.classifier = None

        if model_params[1] == True:
            self.conv1 = StoX_Conv2d(1, 16, stox_params_conv, 5)
            self.linear1 = StoX_Linear(16 * 5 * 5, 256, stox_params_linear)
            self.classifier = StoX_Linear(256, 10, stox_params_linear)

        elif model_params[1] == False:
            self.conv1 = nn.Conv2d(1, 16, 5)
            self.linear1 = nn.Linear(16 * 5 * 5, 256)
            self.classifier = nn.Linear(256, 10)

        else:
            raise ValueError("Check args.stox")

    def forward(self, x):
        out = F.avg_pool2d(x, 2)

        out = self.conv1(out)
        out = F.tanh(out)

        out = F.avg_pool2d(out, 2)
        out = out.view(out.size()[0], -1)

        out = self.linear1(out)
        out = F.tanh(out)

        out = self.classifier(out)

        return out


class cifarNetwork(nn.Module):
    def __init__(self, stox_params_conv, stox_params_linear, model_params):
        super(cifarNetwork, self).__init__()
        self.conv1 = self.conv2 = self.conv3 = self.linear1 = self.classifier = None

        self.conv1 = StoX_Conv2d(3, 8, stox_params_conv, 3)
        self.conv2 = StoX_Conv2d(8, 16, stox_params_conv, 3)
        self.conv3 = StoX_Conv2d(16, 32, stox_params_conv, 3)

        self.linear1 = StoX_Linear(288, 128, stox_params_linear)
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        out = F.avg_pool2d(x, 2)

        out = self.conv1(out)
        out = F.tanh(out)
        out = F.avg_pool2d(out, 2)

        out = self.conv2(out)
        out = F.tanh(out)

        out = self.conv3(out)
        out = F.tanh(out)

        out = out.view(out.size()[0], -1)

        out = self.linear1(out)
        out = F.tanh(out)

        out = self.classifier(out)

        return out


def toy_model_mnist(stox_params_conv, stox_params_linear, model_params):
    return LinearNetwork(stox_params_conv, stox_params_linear, model_params)


def toy_model_cifar10(stox_params_conv, stox_params_linear, model_params):
    return cifarNetwork(stox_params_conv, stox_params_linear, model_params)