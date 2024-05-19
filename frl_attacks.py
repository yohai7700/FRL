from args import args
import torch

def reverse_attack(reputations):
    return torch.sort(reputations, descending=True)[1]

def swap_half_attack(reputations):
    return torch.sort(reputations, descending=True)[1]

def get_frl_attack():
    if args.frl_attack == "reverse":
        return reverse_attack