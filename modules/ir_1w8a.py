import torch.nn as nn
from typing import Optional, Any

from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init
# from torch.nn.modules._functions import SyncBatchNorm as sync_batch_norm
# from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.modules.module import Module
from . import binaryfunction
import torch
from torch import Tensor
import math
# from quantizeInt8 import activation_quantize_fn
from torch.nn.parameter import Parameter


class IRConv2d_conv1(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(IRConv2d_conv1, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                             bias)
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()
        self.bw = 0
        # self.a_quantize_fn = activation_quantize_fn(a_bit=8)

    def forward(self, input):
        w = self.weight
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)

        sw = torch.pow(torch.tensor([2] * bw.size(0)).cuda().float(),
                       (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(
            bw.size(0), 1, 1, 1).detach()

        bw = binaryfunction.BinaryQuantize().apply(bw, self.k, self.t)

        output = torch.round(F.conv2d(input, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups))

        return output, 0, sw


class IRConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(IRConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()
        self.bw = 0

    def forward(self, input):
        w = self.weight
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)

        sw = torch.pow(torch.tensor([2] * bw.size(0)).cuda().float(),
                       (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(
            bw.size(0), 1, 1, 1).detach()

        bw = binaryfunction.BinaryQuantize().apply(bw, self.k, self.t)

        weight_q = bw
        # bw = bw * sw
        output = F.conv2d(input, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)

        return output, 0, sw


class IRConv2d_cim(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(IRConv2d_cim, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                           bias)
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()
        self.bw = 0

    def forward(self, input):
        w = self.weight
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)

        sw = torch.pow(torch.tensor([2] * bw.size(0)).cuda().float(),
                       (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().cuda().float()).view(
            bw.size(0), 1, 1, 1).detach()

        bw = binaryfunction.BinaryQuantize().apply(bw, self.k, self.t)

        weight_q = bw.cuda()
        data = input

        input_one_bit = data % 2
        data = torch.floor(data / 2)

        input_two_bit = data % 2
        data = torch.floor(data / 2)

        input_three_bit = data % 2
        data = torch.floor(data / 2)

        input_four_bit = data % 2

        if self.in_channels == 16:
            self.resu1 = F.conv2d(input_one_bit, weight_q, self.bias, self.stride, self.padding, self.dilation,
                                  self.groups)
            self.resu2 = F.conv2d(input_two_bit, weight_q, self.bias, self.stride, self.padding, self.dilation,
                                  self.groups)
            self.resu3 = F.conv2d(input_three_bit, weight_q, self.bias, self.stride, self.padding, self.dilation,
                                  self.groups)
            self.resu4 = F.conv2d(input_four_bit, weight_q, self.bias, self.stride, self.padding, self.dilation,
                                  self.groups)
            self.resu1 = torch.clamp(self.resu1, -128, 127)
            self.resu2 = torch.clamp(self.resu2, -128, 127)
            self.resu3 = torch.clamp(self.resu3, -128, 127)
            self.resu4 = torch.clamp(self.resu4, -128, 127)

            # bank_max = 128
            # adc_max = 32
            # self.resu1 = torch.clamp(torch.round(self.resu1 / bank_max * adc_max), -adc_max, adc_max - 1) * 4
            # self.resu2 = torch.clamp(torch.round(self.resu2 / bank_max * adc_max), -adc_max, adc_max - 1) * 4
            # self.resu3 = torch.clamp(torch.round(self.resu3 / bank_max * adc_max), -adc_max, adc_max - 1) * 4
            # self.resu4 = torch.clamp(torch.round(self.resu4 / bank_max * adc_max), -adc_max, adc_max - 1) * 4

        elif self.in_channels == 32:
            input_one_bit_1 = input_one_bit[:, :16]
            input_one_bit_2 = input_one_bit[:, 16:]

            input_two_bit_1 = input_two_bit[:, :16]
            input_two_bit_2 = input_two_bit[:, 16:]

            input_three_bit_1 = input_three_bit[:, :16]
            input_three_bit_2 = input_three_bit[:, 16:]

            input_four_bit_1 = input_four_bit[:, :16]
            input_four_bit_2 = input_four_bit[:, 16:]

            weight_q_1 = weight_q[:, :16]
            weight_q_2 = weight_q[:, 16:]

            self.resu1_1 = F.conv2d(input_one_bit_1, weight_q_1, self.bias, self.stride, self.padding, self.dilation,
                                  self.groups)
            self.resu2_1 = F.conv2d(input_two_bit_1, weight_q_1, self.bias, self.stride, self.padding, self.dilation,
                                  self.groups)
            self.resu3_1 = F.conv2d(input_three_bit_1, weight_q_1, self.bias, self.stride, self.padding, self.dilation,
                                  self.groups)
            self.resu4_1 = F.conv2d(input_four_bit_1, weight_q_1, self.bias, self.stride, self.padding, self.dilation,
                                  self.groups)

            self.resu1_2 = F.conv2d(input_one_bit_2, weight_q_2, self.bias, self.stride, self.padding, self.dilation,
                                    self.groups)
            self.resu2_2 = F.conv2d(input_two_bit_2, weight_q_2, self.bias, self.stride, self.padding, self.dilation,
                                    self.groups)
            self.resu3_2 = F.conv2d(input_three_bit_2, weight_q_2, self.bias, self.stride, self.padding, self.dilation,
                                    self.groups)
            self.resu4_2 = F.conv2d(input_four_bit_2, weight_q_2, self.bias, self.stride, self.padding, self.dilation,
                                    self.groups)

            self.resu1_1 = torch.clamp(self.resu1_1, -128, 127)
            self.resu2_1 = torch.clamp(self.resu2_1, -128, 127)
            self.resu3_1 = torch.clamp(self.resu3_1, -128, 127)
            self.resu4_1 = torch.clamp(self.resu4_1, -128, 127)

            self.resu1_2 = torch.clamp(self.resu1_2, -128, 127)
            self.resu2_2 = torch.clamp(self.resu2_2, -128, 127)
            self.resu3_2 = torch.clamp(self.resu3_2, -128, 127)
            self.resu4_2 = torch.clamp(self.resu4_2, -128, 127)

            # bank_max = 128
            # adc_max = 32
            # #
            # self.resu1_1 = torch.clamp(torch.round(self.resu1_1 / bank_max * adc_max), -adc_max, adc_max - 1) * 4
            # self.resu2_1 = torch.clamp(torch.round(self.resu2_1 / bank_max * adc_max), -adc_max, adc_max - 1) * 4
            # self.resu3_1 = torch.clamp(torch.round(self.resu3_1 / bank_max * adc_max), -adc_max, adc_max - 1) * 4
            # self.resu4_1 = torch.clamp(torch.round(self.resu4_1 / bank_max * adc_max), -adc_max, adc_max - 1) * 4
            #
            # self.resu1_2 = torch.clamp(torch.round(self.resu1_2 / bank_max * adc_max), -adc_max, adc_max - 1) * 4
            # self.resu2_2 = torch.clamp(torch.round(self.resu2_2 / bank_max * adc_max), -adc_max, adc_max - 1) * 4
            # self.resu3_2 = torch.clamp(torch.round(self.resu3_2 / bank_max * adc_max), -adc_max, adc_max - 1) * 4
            # self.resu4_2 = torch.clamp(torch.round(self.resu4_2 / bank_max * adc_max), -adc_max, adc_max - 1) * 4

            self.resu1 = self.resu1_1 + self.resu1_2
            self.resu2 = self.resu2_1 + self.resu2_2
            self.resu3 = self.resu3_1 + self.resu3_2
            self.resu4 = self.resu4_1 + self.resu4_2

        else:
            input_one_bit_1 = input_one_bit[:, :16]
            input_one_bit_2 = input_one_bit[:, 16:32]
            input_one_bit_3 = input_one_bit[:, 32:48]
            input_one_bit_4 = input_one_bit[:, 48:]

            input_two_bit_1 = input_two_bit[:, :16]
            input_two_bit_2 = input_two_bit[:, 16:32]
            input_two_bit_3 = input_two_bit[:, 32:48]
            input_two_bit_4 = input_two_bit[:, 48:]

            input_three_bit_1 = input_three_bit[:, :16]
            input_three_bit_2 = input_three_bit[:, 16:32]
            input_three_bit_3 = input_three_bit[:, 32:48]
            input_three_bit_4 = input_three_bit[:, 48:]

            input_four_bit_1 = input_four_bit[:, :16]
            input_four_bit_2 = input_four_bit[:, 16:32]
            input_four_bit_3 = input_four_bit[:, 32:48]
            input_four_bit_4 = input_four_bit[:, 48:]

            weight_q_1 = weight_q[:, :16]
            weight_q_2 = weight_q[:, 16:32]
            weight_q_3 = weight_q[:, 32:48]
            weight_q_4 = weight_q[:, 48:]

            self.resu1_1 = F.conv2d(input_one_bit_1, weight_q_1, self.bias, self.stride, self.padding, self.dilation,
                                    self.groups)
            self.resu2_1 = F.conv2d(input_two_bit_1, weight_q_1, self.bias, self.stride, self.padding, self.dilation,
                                    self.groups)
            self.resu3_1 = F.conv2d(input_three_bit_1, weight_q_1, self.bias, self.stride, self.padding, self.dilation,
                                    self.groups)
            self.resu4_1 = F.conv2d(input_four_bit_1, weight_q_1, self.bias, self.stride, self.padding, self.dilation,
                                    self.groups)

            self.resu1_2 = F.conv2d(input_one_bit_2, weight_q_2, self.bias, self.stride, self.padding, self.dilation,
                                    self.groups)
            self.resu2_2 = F.conv2d(input_two_bit_2, weight_q_2, self.bias, self.stride, self.padding, self.dilation,
                                    self.groups)
            self.resu3_2 = F.conv2d(input_three_bit_2, weight_q_2, self.bias, self.stride, self.padding, self.dilation,
                                    self.groups)
            self.resu4_2 = F.conv2d(input_four_bit_2, weight_q_2, self.bias, self.stride, self.padding, self.dilation,
                                    self.groups)

            self.resu1_3 = F.conv2d(input_one_bit_3, weight_q_3, self.bias, self.stride, self.padding, self.dilation,
                                    self.groups)
            self.resu2_3 = F.conv2d(input_two_bit_3, weight_q_3, self.bias, self.stride, self.padding, self.dilation,
                                    self.groups)
            self.resu3_3 = F.conv2d(input_three_bit_3, weight_q_3, self.bias, self.stride, self.padding, self.dilation,
                                    self.groups)
            self.resu4_3 = F.conv2d(input_four_bit_3, weight_q_3, self.bias, self.stride, self.padding, self.dilation,
                                    self.groups)

            self.resu1_4 = F.conv2d(input_one_bit_4, weight_q_4, self.bias, self.stride, self.padding, self.dilation,
                                    self.groups)
            self.resu2_4 = F.conv2d(input_two_bit_4, weight_q_4, self.bias, self.stride, self.padding, self.dilation,
                                    self.groups)
            self.resu3_4 = F.conv2d(input_three_bit_4, weight_q_4, self.bias, self.stride, self.padding, self.dilation,
                                    self.groups)
            self.resu4_4 = F.conv2d(input_four_bit_4, weight_q_4, self.bias, self.stride, self.padding, self.dilation,
                                    self.groups)

            self.resu1_1 = torch.clamp(self.resu1_1, -128, 127)
            self.resu2_1 = torch.clamp(self.resu2_1, -128, 127)
            self.resu3_1 = torch.clamp(self.resu3_1, -128, 127)
            self.resu4_1 = torch.clamp(self.resu4_1, -128, 127)
            
            self.resu1_2 = torch.clamp(self.resu1_2, -128, 127)
            self.resu2_2 = torch.clamp(self.resu2_2, -128, 127)
            self.resu3_2 = torch.clamp(self.resu3_2, -128, 127)
            self.resu4_2 = torch.clamp(self.resu4_2, -128, 127)
            
            self.resu1_3 = torch.clamp(self.resu1_3, -128, 127)
            self.resu2_3 = torch.clamp(self.resu2_3, -128, 127)
            self.resu3_3 = torch.clamp(self.resu3_3, -128, 127)
            self.resu4_3 = torch.clamp(self.resu4_3, -128, 127)
            
            self.resu1_4 = torch.clamp(self.resu1_4, -128, 127)
            self.resu2_4 = torch.clamp(self.resu2_4, -128, 127)
            self.resu3_4 = torch.clamp(self.resu3_4, -128, 127)
            self.resu4_4 = torch.clamp(self.resu4_4, -128, 127)

            # bank_max = 128
            # adc_max = 32
            #
            # self.resu1_1 = torch.clamp(torch.round(self.resu1_1 / bank_max * adc_max), -adc_max, adc_max - 1) * 4
            # self.resu2_1 = torch.clamp(torch.round(self.resu2_1 / bank_max * adc_max), -adc_max, adc_max - 1) * 4
            # self.resu3_1 = torch.clamp(torch.round(self.resu3_1 / bank_max * adc_max), -adc_max, adc_max - 1) * 4
            # self.resu4_1 = torch.clamp(torch.round(self.resu4_1 / bank_max * adc_max), -adc_max, adc_max - 1) * 4
            #
            # self.resu1_2 = torch.clamp(torch.round(self.resu1_2 / bank_max * adc_max), -adc_max, adc_max - 1) * 4
            # self.resu2_2 = torch.clamp(torch.round(self.resu2_2 / bank_max * adc_max), -adc_max, adc_max - 1) * 4
            # self.resu3_2 = torch.clamp(torch.round(self.resu3_2 / bank_max * adc_max), -adc_max, adc_max - 1) * 4
            # self.resu4_2 = torch.clamp(torch.round(self.resu4_2 / bank_max * adc_max), -adc_max, adc_max - 1) * 4
            #
            # self.resu1_3 = torch.clamp(torch.round(self.resu1_3 / bank_max * adc_max), -adc_max, adc_max - 1) * 4
            # self.resu2_3 = torch.clamp(torch.round(self.resu2_3 / bank_max * adc_max), -adc_max, adc_max - 1) * 4
            # self.resu3_3 = torch.clamp(torch.round(self.resu3_3 / bank_max * adc_max), -adc_max, adc_max - 1) * 4
            # self.resu4_3 = torch.clamp(torch.round(self.resu4_3 / bank_max * adc_max), -adc_max, adc_max - 1) * 4
            #
            # self.resu1_4 = torch.clamp(torch.round(self.resu1_4 / bank_max * adc_max), -adc_max, adc_max - 1) * 4
            # self.resu2_4 = torch.clamp(torch.round(self.resu2_4 / bank_max * adc_max), -adc_max, adc_max - 1) * 4
            # self.resu3_4 = torch.clamp(torch.round(self.resu3_4 / bank_max * adc_max), -adc_max, adc_max - 1) * 4
            # self.resu4_4 = torch.clamp(torch.round(self.resu4_4 / bank_max * adc_max), -adc_max, adc_max - 1) * 4

            self.resu1 = self.resu1_1 + self.resu1_2 + self.resu1_3 + self.resu1_4
            self.resu2 = self.resu2_1 + self.resu2_2 + self.resu2_3 + self.resu2_4
            self.resu3 = self.resu3_1 + self.resu3_2 + self.resu3_3 + self.resu3_4
            self.resu4 = self.resu4_1 + self.resu4_2 + self.resu4_3 + self.resu4_4

        result = self.resu1 + self.resu2 * 2 + self.resu3 * 4 + self.resu4 * (-8)
        return result, sw


class IRlinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super(IRlinear, self).__init__(in_features, out_features, bias=False)
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()

    def forward(self, input, Tmax=[1.0, 0.0]):
        w = self.weight
        b = self.bias
        x = input
        Tmax = torch.max(input).detach()  # Tmax[0]  # torch.max(input).detach()#
        Tmin = torch.min(input).detach()  # Tmax[1]  # torch.min(input).detach()#
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1)
        sw = torch.pow(torch.tensor([2] * bw.size(0)).cuda().float(),
                       (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(
            bw.size(0), 1).detach()

        bw = binaryfunction.BinaryQuantize().apply(bw, self.k, self.t)
        # bw = bw * sw

        # activation
        # T = torch.max(torch.abs(Tmin), torch.abs(Tmax))
        # T = torch.clamp(T, 1e-10, 255.)
        # # quan?
        # x = torch.clamp(x, 0 - T, T)
        # x_s = x / T
        # activation_q = binaryfunction.qfn().apply(x_s, 4)
        # activation_q = activation_q * T
        output = torch.round(F.linear(input, bw, self.bias))
        return output
