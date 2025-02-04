import torch
import torch.nn as nn
import torch.nn.functional as F
from src.InputQuantization import *
from src.MTJQuantization import *
from src.WeightQuantization import *
from src.debug import tensor_stats, print_num_unique_vals, plot_tensor_hist
import random


def get_chunks(in_channels, subarray_size):
    return (in_channels / subarray_size).__ceil__()


class StoX_MTJ(nn.Module):
    def __init__(self, channels, sensitivity):
        super(StoX_MTJ, self).__init__()
        self.bn = nn.BatchNorm1d(channels)
        # self.constrictor = nn.Parameter(torch.tensor(1.0))
        self.sensitivity = sensitivity

    def forward(self, input_tens):
        # normed_input = input_tens / self.constrictor
        normed_input = self.bn(input_tens).clamp(-1, 1)
        out = MTJInstance().apply(normed_input, self.sensitivity)
        return out


class StoX_Linear(nn.Module):
    def __init__(self, in_channels, out_channels, stox_params):
        super(StoX_Linear, self).__init__()
        # StoX_Params order -> [a_bits, w_bits, a_stream_width, w_slice_width, subarray_size, sensitivity, time_steps]

        # Define crossbar-related parameters
        self.num_subarrays = get_chunks(in_channels, subarray_size=stox_params[4])
        self.w_bits = stox_params[1]
        self.w_bits_per_slice = stox_params[3]
        self.w_slices = int(self.w_bits / self.w_bits_per_slice)
        self.a_bits = stox_params[0]
        self.a_bits_per_stream = stox_params[2]
        self.a_slices = int(self.a_bits / self.a_bits_per_stream)

        # Define MTJ-related parameters
        self.MTJ = StoX_MTJ(out_channels, sensitivity=stox_params[5])
        self.sensitivity = stox_params[5]
        self.iterations = stox_params[-1]

        # Initialize layer weights
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        nn.init.kaiming_normal_(self.weight)

    def StoX_LinearOp(self, input_activation, weights):
        input_list = torch.chunk(input_activation, chunks=self.num_subarrays, dim=1)
        weight_list = torch.chunk(weights, chunks=self.num_subarrays, dim=1)

        output = 0

        for i, working_weight in enumerate(weight_list):
            working_input = input_list[i]
            linear_temp = F.linear(working_input, working_weight)
            linear_temp = torch.stack([linear_temp] * self.iterations, -1)
            output += self.MTJ(linear_temp)

        output = output.sum(-1) / (self.num_subarrays * self.iterations)

        return output

    def forward(self, inputs):
        w = self.weight

        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1)
        qw = WeightQuantize().apply(bw, self.w_bits_per_slice)

        if self.a_bits == 1:
            qa = quantize_STE_floor_ceil(inputs, self.a_bits)
        elif (self.a_bits > 1) and (self.a_slices == 1):
            qa = quantize_STE_round(inputs, self.a_bits)
        else:
            exit()

        linear_stoch_out = self.StoX_LinearOp(qa, qw)

        return linear_stoch_out


class StoX_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stox_params, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=None):
        super(StoX_Conv2d, self).__init__()
        # StoX_Params order -> [a_bits, w_bits, a_stream_width, w_slice_width, subarray_size, sensitivity, time_steps]

        # Define crossbar array parameters
        self.num_subarrays = get_chunks(in_channels * (kernel_size ** 2), subarray_size=stox_params[4])
        self.w_bits = stox_params[1]
        self.w_bits_per_slice = stox_params[3]
        self.w_slices = int(self.w_bits / self.w_bits_per_slice)
        self.a_bits = stox_params[0]
        self.a_bits_per_stream = stox_params[2]
        self.a_slices = int(self.a_bits / self.a_bits_per_stream)

        # Define the row of MTJs at the crossbar's output
        self.MTJ = StoX_MTJ(out_channels * self.w_slices, sensitivity=stox_params[5])
        self.sensitivity = stox_params[5]
        self.iterations = stox_params[-1]

        # Define standard convolution parameters
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.bias = bias
        self.kernel_size = kernel_size

        # Initialize layer weights
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight)

    def StoX_hardware_Conv(self, image_map, filter_weights, bias, stride, padding, dilation, groups):
        flattened_weights = torch.flatten(filter_weights, 1)
        batch_size = image_map.size(0)

        kernel_list = torch.chunk(image_map, chunks=self.num_subarrays, dim=1)
        weight_list = torch.chunk(flattened_weights, chunks=self.num_subarrays, dim=1)

        output = 0
        for i, working_weight in enumerate(weight_list):
            working_kernel = kernel_list[i].transpose(-2, -1)
            # Size = [batches, out_Chann * w_slices, tot_Pix * a_slices]
            linear_temp = F.linear(working_kernel, working_weight).transpose(-2, -1)
            linear_temp = torch.concat([linear_temp] * self.iterations, 0)
            output = output + self.MTJ(linear_temp)

        output = torch.stack(output.split(batch_size, 0), -1).sum(-1)
        output = output / (self.num_subarrays * self.iterations)

        out_pixels = int((output.size(dim=-1)) ** 0.5)  # Image size to map back to
        result = F.fold(output, (out_pixels, out_pixels), (1, 1))  # Fold result into image
        return result

    def forward(self, inputs):  # Input dims = [out_channels, in_channels, kernel_height, kernel_width]
        w = self.weight
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)
        qw = WeightQuantize().apply(bw, self.w_bits_per_slice)

        a = F.unfold(inputs, self.kernel_size, self.dilation, self.padding, self.stride)

        if self.a_bits == 1:
            qa = quantize_STE_floor_ceil(a, self.a_bits)
        elif (self.a_bits > 1) and (self.a_slices == 1):
            qa = quantize_STE_round(a, self.a_bits)
        else:
            exit()

        conv_stoch_out = self.StoX_hardware_Conv(qa, qw, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return conv_stoch_out
