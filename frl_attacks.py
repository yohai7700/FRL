from args import args
import torch

def reverse_attack(reputations):
    return torch.sort(reputations, descending=True)[1]

def swap_half_attack(reputations):
    ranking = torch.sort(reputations)[1]
    half_len = len(ranking) // 2
    first_half = ranking[:half_len]
    second_half = ranking[half_len:]
    return torch.cat([second_half, first_half])

def swap_half_reverse_attack(reputations):
    ranking = torch.sort(reputations)[1]
    half_len = len(ranking) // 2
    first_half = ranking[:half_len]
    second_half = torch.flip(ranking[half_len:], 1)
    return torch.cat([second_half, first_half])

def get_frl_attack():
    if args.frl_attack == "reverse":
        return reverse_attack
    if args.frl_attack == "swap_half":
        return swap_half_attack