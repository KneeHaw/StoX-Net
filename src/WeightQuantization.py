import torch
from torch.autograd import Function


class WeightQuantize(Function):
    @staticmethod
    def forward(ctx, input_tens, bits):
        if bits > 1:
            input_tens = input_tens * (2 ** (bits - 1))
            out = torch.where(input_tens > 0, input_tens.ceil(), input_tens)
            out = torch.where(input_tens < 0, input_tens.floor(), out)
            out = torch.where(input_tens == 0, input_tens, out) / (2 ** (bits - 1))
        elif bits == 1:
            out = torch.where(input_tens < 0, torch.floor(input_tens), input_tens)
            out = torch.where(out > 0, torch.ceil(out), out)
            out = torch.where(input_tens == 0, input_tens, out)
        else:
            raise ValueError("Weight bits cannot be negative")

        out = torch.clamp(out, -1, 1)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def gen_weight_vector_and_sum(tensor, slices):
    vector = (2 ** torch.arange(0, slices, 1, device='cuda'))
    tensor_sum = torch.stack(torch.tensor_split(tensor, slices, dim=1), dim=-1) * vector / (2 ** slices - 1)
    tensor_sum = tensor_sum.sum(-1)
    return tensor_sum