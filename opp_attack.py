import torch

def swap_half(tensor):
    half_len = (len(tensor) // 2) - 20
    first_half = tensor[:half_len]
    second_half = tensor[half_len:]
    return torch.cat([second_half, first_half])