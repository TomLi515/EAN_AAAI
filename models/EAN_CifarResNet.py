import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch
import sys
"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter

class EAN(nn.Module):
    def __init__(self, channels, groups=64, mode='l2'):
        super(EAN, self).__init__()
        self.groups = groups
        self.gn = nn.GroupNorm(channels // groups, channels // groups, affine=True)
        self.sigmoid = nn.Sigmoid()
        self.gamma = nn.Parameter(torch.zeros(1, channels // groups, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels // groups, 1, 1))
        self.epsilon = 1e-5
        self.mode = mode

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b * self.groups, -1, h, w)
        xs = self.gn(x)
        weight = self.gn.weight
        weight = weight.view(1, -1, 1, 1)
        if self.mode == 'l2':
            weight1 = (weight.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
            norm = self.gamma * (weight / weight1)
        elif self.mode == 'l1':
            weight2 = torch.abs(weight).mean(dim=1, keepdim=True) + self.epsilon
            norm = self.gamma * (weight / weight2)
        else:
            print('Unknown mode!')
            sys.exit()
        out = x * self.sigmoid(xs * norm + self.beta)
        out = out.view(b, -1, h, w)

        return out



class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            EAN(out_channels * BasicBlock.expansion),
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
            EAN(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class resnet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


# def ResNet18():
#     """ return a ResNet 18 object
#     """
#     return resnet(BasicBlock, [2, 2, 2, 2])
#
# def ResNet34():
#     """ return a ResNet 34 object
#     """
#     return resnet(BasicBlock, [3, 4, 6, 3])
#
# def ResNet50():
#     """ return a ResNet 50 object
#     """
#     return resnet(BottleNeck, [3, 4, 6, 3])
#
# def ResNet101():
#     """ return a ResNet 101 object
#     """
#     return resnet(BottleNeck, [3, 4, 23, 3])
#
# def ResNet152():
#     """ return a ResNet 152 object
#     """
#     return resnet(BottleNeck, [3, 8, 36, 3])

# 50
def ResNet(num_classes=100):
    model = resnet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes)
    return model


if __name__ == '__main__':
    batch = 1
    inplanes = 3
    outplanes = 100
    h, w = 32, 32

    # from .model_summary import model_summary

    model = ResNet(outplanes)
    x = torch.rand((batch, inplanes, h, w))
    model(x)
    print(model)
    # print(model_summary(model, (inplanes, h, w)))

    from thop import clever_format, profile

    print(clever_format(profile(model, inputs=(torch.rand(1, inplanes, h, w),))))
