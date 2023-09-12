'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import numpy as np
import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
from torch.autograd import Variable
from modules import binaryfunction

from modules import ir_1w8a

torch.set_printoptions(profile="full")
torch.set_printoptions(precision=4, sci_mode=False)
__all__ = ['resnet12_1w8a']


def _weights_init(m):
    classname = m.__class__.__name__
    print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock_1w8a_q(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, idx=0, layer_idx=0, option='A'):
        super(BasicBlock_1w8a_q, self).__init__()
        self.conv1 = ir_1w8a.IRConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = ir_1w8a.IRConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()
        self.layer_idx = layer_idx
        self.idx = idx
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    ir_1w8a.IRConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def my_quantize_conv(self, input, prec):
        x = input
        Tmax = torch.max(input).detach()
        Tmin = torch.min(input).detach()
        T = torch.max(torch.abs(Tmin), torch.abs(Tmax))
        T = torch.clamp(T, 1e-10, 255.)

        x = torch.clamp(x, 0 - T, T)
        x_s = x / T
        if Tmin >= 0:
            activation_q = binaryfunction.qfn().apply(x_s, prec)
        else:
            activation_q = binaryfunction.qfn().apply(x_s * 0.5, prec)
        return activation_q

    def my_quantize(self, input, prec):
        x = input
        Tmax = torch.max(input).detach()
        Tmin = torch.min(input).detach()
        T = torch.max(torch.abs(Tmin), torch.abs(Tmax))
        T = torch.clamp(T, 1e-10, 255.)
        x = torch.clamp(x, 0 - T, T)
        x_s = x / T
        activation_q = binaryfunction.qfn().apply(x_s, prec)
        activation_q = activation_q * T
        return activation_q

    def my_quantize_7(self, input, prec):
        x = input

        Tmax = torch.max(input).detach()
        Tmin = torch.min(input).detach()
        T = torch.max(torch.abs(Tmin), torch.abs(Tmax))
        # T = torch.clamp(T, 1e-10, 255.)
        x = torch.clamp(x, 0 - T, T)
        x_s = x / T
        activation_q = binaryfunction.qfn().apply(x_s, prec)
        n = float(2 ** prec - 1)
        activation_q = activation_q * n
        # activation_q = activation_q * T

        return activation_q, T

    def my_bn1(self, input, sw_conv1, err, input_channel):
        global var, mean
        if self.training:
            y = input
            y = y.permute(1, 0, 2, 3)
            y = y.contiguous().view(y.shape[0], -1)
            mean = y.mean(1).detach()
            var = y.var(1).detach()
            self.bn1.running_mean = self.bn1.momentum * self.bn1.running_mean + (1 - self.bn1.momentum) * mean
            self.bn1.running_var = self.bn1.momentum * self.bn1.running_var + (1 - self.bn1.momentum) * var
        else:  # BN2?2??¨¹D?
            mean = self.bn1.running_mean
            var = self.bn1.running_var
        std = torch.sqrt(var + self.bn1.eps)
        weight = self.bn1.weight / std
        bias = self.bn1.bias - weight * mean
        weight = weight.view(input.shape[1], 1)

        p3d = (0, input.shape[1] - 1)
        weight = F.pad(weight, p3d, 'constant', 0)
        for i in range(input.shape[1]):
            weight[i][i] = weight[i][0]
            if i > 0:
                weight[i][0] = 0

        weight = weight.view(input.shape[1], input.shape[1], 1, 1)

        T_a = 7 * 3 * 3 * 64
        T_w = 1
        activation_bit = 10
        activation_k = 2 ** activation_bit - 1

        bw, T_w = self.my_quantize_7(weight, 3)
        activation_q = torch.floor(torch.round(input / 4) * sw_conv1)

        bb = torch.round(self.my_quantize(bias, (activation_bit + 4)) * 7 * activation_k / T_a * 7 / T_w)

        out = torch.round(F.conv2d(activation_q, bw, bb, stride=1, padding=0))

        return out, T_a, T_w

    def my_bn2(self, input, sw_conv2, err2, input_channel):
        global var2, mean2
        if self.training:
            y2 = input
            y2 = y2.permute(1, 0, 2, 3)
            y2 = y2.contiguous().view(y2.shape[0], -1)
            mean2 = y2.mean(1).detach()
            var2 = y2.var(1).detach()
            self.bn2.running_mean = self.bn2.momentum * self.bn2.running_mean + (1 - self.bn2.momentum) * mean2
            self.bn2.running_var = self.bn2.momentum * self.bn2.running_var + (1 - self.bn2.momentum) * var2
        else:  # BN2?2??¨¹D?
            mean2 = self.bn2.running_mean
            var2 = self.bn2.running_var
        std2 = torch.sqrt(var2 + self.bn2.eps)
        weight2 = self.bn2.weight / std2
        bias2 = self.bn2.bias - weight2 * mean2
        weight2 = weight2.view(input.shape[1], 1)
        p3d2 = (0, input.shape[1] - 1)
        weight2 = F.pad(weight2, p3d2, 'constant', 0)
        for i in range(input.shape[1]):
            weight2[i][i] = weight2[i][0]
            if i > 0:
                weight2[i][0] = 0
        weight2 = weight2.view(input.shape[1], input.shape[1], 1, 1)
        T_w2 = 1
        T_a2 = 7 * 3 * 3 * 64
        bw2, T_w2 = self.my_quantize_7(weight2, 3)

        activation_bit = 10
        activation_k = 2 ** activation_bit - 1

        activation_q2 = torch.floor(torch.round(input / 4) * sw_conv2)
        bb2 = torch.round(self.my_quantize(bias2, (activation_bit + 4)) * 7 * activation_k / T_a2 * 7 / T_w2)

        out = torch.round(F.conv2d(activation_q2, bw2, bb2, stride=1, padding=0))

        return out, T_a2, T_w2

    def forward(self, x):
        if (self.layer_idx == 1) and ((self.idx == 1) or (self.idx == 2)):
            out = x
        else:
            # ---------------------conv1-----------------------
            x_cim = x
            cim_conv = ir_1w8a.IRConv2d_cim(self.conv1.in_channels, self.conv1.out_channels, kernel_size=(3, 3),
                                            stride=self.conv1.stride, padding=self.conv1.padding, bias=False)
            cim_conv.weight = self.conv1.weight
            out, sw_conv1 = cim_conv(x_cim)
            sw_conv1 = sw_conv1.view(1, sw_conv1.shape[0], 1, 1)
            # ---------------------bn1-----------------------
            out, T_a, T_w = self.my_bn1(out, sw_conv1, 0, self.conv1.in_channels)
            # ---------------------shortcut1-----------------------
            short_cut = torch.round(1023 / T_a * 7 / T_w)
            out += torch.round(self.shortcut(x * short_cut))
            # ---------------------hardtanh1-----------------------
            T_hardtanh = torch.round(7 * 1023 / T_a * 7 / T_w).float()
            out = torch.round(F.hardtanh(out, min_val=-T_hardtanh, max_val=T_hardtanh) / T_hardtanh * 7)
        if (self.layer_idx == 1) and ((self.idx == 1) or (self.idx == 2)):
            out = out
        else:
            x1 = out
            x1_cim = x1
            # ---------------------conv2-----------------------
            cim_conv2 = ir_1w8a.IRConv2d_cim(self.conv2.in_channels, self.conv2.out_channels, kernel_size=(3, 3),
                                             stride=self.conv2.stride, padding=self.conv2.padding, bias=False)
            cim_conv2.weight = self.conv2.weight
            out, sw_conv2 = cim_conv2(x1_cim)
            sw_conv2 = sw_conv2.view(1, sw_conv2.shape[0], 1, 1)
            # ---------------------bn2-----------------------
            out, T_a2, T_w2 = self.my_bn2(out, sw_conv2, 0, self.conv2.in_channels)
            # ---------------------add2-----------------------
            short_cut = torch.round(1023 / T_a2 * 7 / T_w2)
            out = torch.round(out + (x1 * short_cut))
            # ---------------------hardtanh2-----------------------
            T_hardtanh2 = torch.round(7 * 1023 / T_a2 * 7 / T_w2).float()
            out = torch.round(F.hardtanh(out, min_val=-T_hardtanh2, max_val=T_hardtanh2) / T_hardtanh2 * 7)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = ir_1w8a.IRConv2d_conv1(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, layer_idx=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, layer_idx=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, layer_idx=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.linear = ir_1w8a.IRlinear(64, num_classes, bias=False)
        self.apply(_weights_init)
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()

    def _make_layer(self, block, planes, num_blocks, stride, layer_idx):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        idx = 0
        for stride in strides:
            idx = idx + 1
            layers.append(block(self.in_planes, planes, stride, idx, layer_idx))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def my_quantize(self, input, prec):
        x = input
        Tmax = torch.max(input).detach()
        Tmin = torch.min(input).detach()
        T = torch.max(torch.abs(Tmin), torch.abs(Tmax))
        T = torch.clamp(T, 1e-10, 255.)
        x = torch.clamp(x, 0 - T, T)
        x_s = x / T
        activation_q = binaryfunction.qfn().apply(x_s, prec)
        activation_q = activation_q * T
        return activation_q

    def my_quantize_conv1(self, input, prec):
        x = input
        Tmax = torch.max(input).detach()
        Tmin = torch.min(input).detach()
        T = torch.max(torch.abs(Tmin), torch.abs(Tmax))
        T = torch.clamp(T, 1e-10, 255.)
        x = torch.clamp(x, 0 - T, T)
        x_s = x / T
        activation_q = binaryfunction.qfn().apply(x_s, prec)
        # activation_q = activation_q * T
        return activation_q

    def my_quantize_7(self, input, prec):
        x = input
        Tmax = torch.max(input).detach()
        Tmin = torch.min(input).detach()
        T = torch.max(torch.abs(Tmin), torch.abs(Tmax))
        x = torch.clamp(x, 0 - T, T)
        x_s = x / T
        activation_q = binaryfunction.qfn().apply(x_s, prec)
        n = float(2 ** prec - 1)
        activation_q = torch.round(activation_q * n)

        return activation_q, T

    def my_bn(self, input, T, sw_conv1, err):
        global var, mean
        if self.training:
            y = input
            y = y.permute(1, 0, 2, 3)
            y = y.contiguous().view(y.shape[0], -1)
            mean = y.mean(1).detach()
            var = y.var(1).detach()
            self.bn1.running_mean = self.bn1.momentum * self.bn1.running_mean + (1 - self.bn1.momentum) * mean
            self.bn1.running_var = self.bn1.momentum * self.bn1.running_var + (1 - self.bn1.momentum) * var
        else:
            mean = self.bn1.running_mean
            var = self.bn1.running_var
        std = torch.sqrt(var + self.bn1.eps)
        weight = self.bn1.weight / std
        bias = self.bn1.bias - weight * mean
        weight = weight.view(input.shape[1], 1)

        p3d = (0, input.shape[1] - 1)
        weight = F.pad(weight, p3d, 'constant', 0)
        for i in range(input.shape[1]):
            weight[i][i] = weight[i][0]
            if i > 0:
                weight[i][0] = 0
        weight = weight.view(input.shape[1], input.shape[1], 1, 1)

        T_a = 7 * 9 * 3

        bw, T_w = self.my_quantize_7(weight, 3)

        # activation_q_test = torch.round(torch.floor(torch.round(input * 1023 / T_a) * sw_conv1))
        activation_q_test = torch.round(input * 1023 / T_a)
        bb = torch.round(self.my_quantize(bias, 14) * 7 / T_w * 7 * 1023 / T_a)

        # for i in range(16):
        #     print(bw.squeeze()[i,i])
        # exit()
        out = torch.round(F.conv2d(activation_q_test, bw, bb, stride=1, padding=0))

        return out, T_a, T_w

    def my_bn2(self, input):
        global var2, mean2
        if self.training:
            y2 = input
            y2 = y2.permute(1, 0, 2, 3)
            y2 = y2.contiguous().view(y2.shape[0], -1)
            mean2 = y2.mean(1).detach()
            var2 = y2.var(1).detach()
            self.bn2.running_mean = \
                self.bn2.momentum * self.bn2.running_mean + \
                (1 - self.bn2.momentum) * mean2
            self.bn2.running_var = \
                self.bn2.momentum * self.bn2.running_var + \
                (1 - self.bn2.momentum) * var2
        else:
            mean2 = self.bn2.running_mean
            var2 = self.bn2.running_var
        std2 = torch.sqrt(var2 + self.bn2.eps)
        weight2 = self.bn2.weight / std2
        bias2 = self.bn2.bias - weight2 * mean2
        weight2 = weight2.view(input.shape[1], 1)

        p3d2 = (0, input.shape[1] - 1)
        weight2 = F.pad(weight2, p3d2, 'constant', 0)
        for i in range(input.shape[1]):
            weight2[i][i] = weight2[i][0]
            if i > 0:
                weight2[i][0] = 0
        weight2 = weight2.view(input.shape[1], input.shape[1], 1, 1)

        T_a2 = 448

        weight2, T_w2 = self.my_quantize_7(weight2, 3)

        activation_q2 = torch.round(input * 1023 / T_a2)

        bb2 = torch.round(self.my_quantize(bias2, 14) * 7 * 64 * 1023 / T_a2 * 7 / T_w2)

        out = torch.round(F.conv2d(activation_q2, weight2, bb2, stride=1, padding=0))
        return out, T_a2, T_w2

    def forward(self, x):

        # x1, T = self.my_quantize_7(x, 3)
        # out, err, sw_conv1 = self.conv1(x1)
        #
        # err = 0
        # sw_conv1 = sw_conv1.view(1, 16, 1, 1)
        #
        # out, T_a, T_w = self.my_bn(out, T, sw_conv1, err)
        # # print(out)
        # # exit()
        # out = torch.round(F.hardtanh(out, min_val=-(7 / T_w * 7 * 1023 / T_a), max_val=(7 / T_w * 7 * 1023 / T_a)) / (
        #         7 / T_w * 7 * 1023 / T_a) * 7)
        out = x
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out0 = out.view(64, 8, 8)
        out0 = torch.sum(out0, dim=2)
        out0 = torch.sum(out0, dim=1)
        out = torch.round(out0.view(1, 64, 1, 1))

        out, T_a2, T_w2 = self.my_bn2(out)

        out = torch.round(F.hardtanh(out, min_val=-(7 * 64 * 1023 / T_a2 * 7 / T_w2),
                                     max_val=(7 * 64 * 1023 / T_a2 * 7 / T_w2)))
        # out = torch.round(out * 7 / (7 * 64 * 1023 / T_a2 * 7 / T_w2))
        out = torch.round(out / (7 * 64 * 1023 / T_a2 / T_w2))
        out = out.view(out.size(0), -1)

        out = torch.round(self.linear(out))

        return out


def resnet12_1w8a():
    return ResNet(BasicBlock_1w8a_q, [2, 2, 2])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params = total_params + np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
