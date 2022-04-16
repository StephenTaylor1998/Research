import torch
from torch import nn
from torch._six import with_metaclass
from torch.autograd import Function


class LinearFunction(Function):

    @staticmethod
    def forward(ctx, input, weight, bias=None):

        return input

    @staticmethod
    def backward(ctx, grad_output):

        return grad_input, grad_weight