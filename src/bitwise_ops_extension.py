import torch
from torch.autograd import Function


def bit_expansion(tensor, bits):
    magic_number = 2 ** bits - 1

    # Constrain tensor to integer range of bits
    tensor = torch.round(tensor * magic_number).clamp(0, magic_number)

    # Create base 2 cutoff vector for bitwise conversion
    arranged = (1 / (2 ** torch.arange(bits, device='cuda')).unsqueeze(0))

    # Convert to binary representation, MSB at dim 0?
    bit_stream = torch.floor(torch.matmul(tensor.unsqueeze(-1), arranged).fmod(2))

    # Reshape to place new dims in existing channels
    bit_stream = bit_stream.reshape(bit_stream.size(0), bit_stream.size(1), -1)

    return bit_stream

class bitwise_expansion(Function):
    @staticmethod
    def forward(ctx, tensor, bits):
        ctx.save_for_backward(tensor, bits)
        magic_number = 2 ** bits - 1

        # Constrain tensor to integer range of bits
        tensor = torch.round(tensor).clamp(0, 2 ** bits - 1)

        # Create base 2 cutoff vector for bitwise conversion
        arranged = (1 / (2 ** torch.arange(bits, device='cuda')).unsqueeze(0))

        # Convert to binary representation, MSB at dim 0?
        bit_stream = torch.floor(torch.matmul(tensor.clamp(0, magic_number).unsqueeze(-1), arranged).fmod(2))

        # Reshape to place new dims in existing channels
        bit_stream = bit_stream.reshape(bit_stream.size(0), bit_stream.size(1), -1)

        return bit_stream

    @staticmethod
    def backward(ctx, grad_output):
        tensor, bits,  = ctx.saved_tensors

        grad_input = (grad_output.split(4, -1) * (2 ** torch.arange(bits, device='cuda')).unsqueeze(0)).sum(-1)
        return grad_input, None, None


class sign_func(Function):
    @staticmethod
    def forward(ctx, input_tensor):
        ctx.save_for_backward(input_tensor)
        output = torch.sign(input_tensor)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, = ctx.saved_tensors
        grad_input = torch.tanh(input_tensor)
        # Example: return grad_output for STE (y = x, derivative 1 at all points)
        # Nones are for passing arguments
        return grad_input, None
