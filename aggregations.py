import torch

def get_voted_ranking_by_sum(rankings):
    sum_args_sorts=torch.sum(rankings, 0)
    idxx=torch.sort(sum_args_sorts)[1]
    return idxx

def get_voting_mechanism(method = "sum"):
    return get_voted_ranking_by_sum