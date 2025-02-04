import torch
from torch.autograd import Function
import torch.nn.functional as F

from src.debug import tensor_stats


class MTJInstance(Function):
    @staticmethod
    def forward(ctx, input_tens, sensitivity):

        input_tens_sigmoid = torch.sigmoid(sensitivity * input_tens)
        rand_tens = torch.rand_like(input_tens_sigmoid, device='cuda:0')
        mask1 = input_tens_sigmoid > rand_tens
        out = mask1.type(torch.float32) - (1 - mask1.type(torch.float32))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None
