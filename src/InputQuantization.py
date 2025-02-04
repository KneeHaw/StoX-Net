import torch
import time


def quantize_STE_floor_ceil(input_tens, bits):
    magic_number_input = 2 ** bits - 1
    with torch.no_grad():
        temp1 = input_tens * magic_number_input
        temp = torch.where(temp1 > 0, torch.ceil(temp1), temp1)
        temp = torch.where(temp1 < 0, torch.floor(temp), temp)
        temp = torch.where(temp1 == 0, 0, temp)
        temp = temp / magic_number_input
    out = input_tens + temp.detach() - input_tens.detach()
    out = torch.clamp(out, -1, 1)
    return out


def quantize_STE_round(input_tens, bits):
    magic_number_input = 2 ** bits - 1
    out = input_tens + ((input_tens * magic_number_input).round() / magic_number_input).detach() - input_tens.detach()
    out = torch.clamp(out, -1, 1)
    return out


def gen_image_vector_and_sum(tensor, bits):
    vector = (2 ** torch.arange(bits, device='cuda'))
    tensor_sum = torch.stack(torch.split(tensor, bits, dim=-1)) * vector / (2 ** bits - 1)
    tensor_sum = tensor_sum.permute(1, 2, 0, 3).sum(-1)
    return tensor_sum
