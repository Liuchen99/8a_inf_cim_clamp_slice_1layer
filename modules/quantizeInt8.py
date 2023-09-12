import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def uniform_quantize(k):
    class qfn(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            if k == 32:
                out = input
            elif k == 1:
                out = torch.sign(input)
            else:
                n = float(2 ** k - 1)
                out = torch.round(input * n) / n
            return out

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            return grad_input

    return qfn().apply


class weight_quantize_fn(nn.Module):
    def __init__(self, w_bit):
        super(weight_quantize_fn, self).__init__()
        assert w_bit <= 8 or w_bit == 32
        self.w_bit = w_bit
        self.uniform_q = uniform_quantize(k=w_bit)

    def forward(self, x):
        if self.w_bit == 32:
            weight_q = x
        elif self.w_bit == 1:
            E = torch.mean(torch.abs(x)).detach()
            weight_q = self.uniform_q(x / E) * E
        else:
            weight = torch.tanh(x)#-1~1
            max_w = torch.max(torch.abs(weight)).detach()
            weight = weight / max_w / 2    #-0.5~0.5
            weight_q = self.uniform_q(weight) #q:-128~127
        return weight_q


class activation_quantize_fn(nn.Module):
    def __init__(self, a_bit):
        super(activation_quantize_fn, self).__init__()
        #assert a_bit <= 8 or a_bit == 32
        self.a_bit = a_bit
        self.uniform_q = uniform_quantize(k=a_bit)

    def forward(self, x, Tmax, Tmin):
        if self.a_bit == 32:
            activation_q = x
        else:
                               
            T = torch.max(torch.abs(Tmin), torch.abs(Tmax))
            T = torch.clamp(T, 1e-10, 255.)
            
            x = torch.clamp(x, 0-T, T) 
            
            x_s = x / T
            
            if(Tmin>=0):
                activation_q = self.uniform_q(x_s)  #0~255
            else:
                activation_q = self.uniform_q(x_s*0.5) ##-128~127
            
        return activation_q


def conv2d_Q_fn(w_bit=8, a_bit=8):
    class Conv2d_Q(nn.Conv2d):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
            super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)
            self.w_bit = w_bit
            self.w_quantize_fn = weight_quantize_fn(w_bit=w_bit)
            self.a_quantize_fn = activation_quantize_fn(a_bit=a_bit)

        def forward(self, input, Tmax=[1.0,0.0],q_en = True, order=None):
            act_max = Tmax[0]  # torch.max(input).detach()#
            act_min = Tmax[1]  # torch.min(input).detach()#
        
            weight_q = self.w_quantize_fn(self.weight)
            if q_en:
                input_q = self.a_quantize_fn(input, act_max, act_min)
            else:
                input_q = input
      
            return F.conv2d(input_q, weight_q, self.bias, self.stride,
                      self.padding, self.dilation, self.groups), torch.cat([(torch.max(input).detach()).unsqueeze(0), (torch.min(input).detach()).unsqueeze(0)], 0)

    return Conv2d_Q
    
def ConvTranspose2d_Q_fn(w_bit=8, a_bit=8):
    class ConvTranspose2d_Q(nn.ConvTranspose2d):
        def __init__(self, in_channels, out_channels, kernel_size=2, stride=1,
                 padding=0, output_padding=0, dilation=1, groups=1, bias=True):
            super(ConvTranspose2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, output_padding, dilation, groups, bias)
            self.w_bit = w_bit
            self.a_bit = a_bit
            self.w_quantize_fn = weight_quantize_fn(w_bit=w_bit)
            self.a_quantize_fn = activation_quantize_fn(a_bit=a_bit)

        def forward(self, input, Tmax=[1.0,0.0], order=None):
            act_max = Tmax[0]  # torch.max(input).detach()#
            act_min = Tmax[1]  # torch.min(input).detach()#

            weight_q = self.w_quantize_fn(self.weight)
            input_q = self.a_quantize_fn(input, act_max, act_min)
            
            return F.conv_transpose2d(input_q, weight_q, self.bias, self.stride,
                      self.padding, self.output_padding, self.groups, self.dilation), torch.cat([(torch.max(input).detach()).unsqueeze(0), (torch.min(input).detach()).unsqueeze(0)], 0)

    return ConvTranspose2d_Q

