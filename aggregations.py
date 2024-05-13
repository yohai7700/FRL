import torch
import ranky as rk

def get_voted_ranking_by_sum(rankings):
    sum_args_sorts=torch.sum(rankings, 0)
    idxx=torch.sort(sum_args_sorts)[1]
    return idxx

def aggregate_ranking_with_trim(rankings):
    fraction = 0.2
    ranking = get_voted_ranking_by_sum(rankings)
    distances = ((rankings - ranking) ** 2).sum(axis = 1)
    sorted_indices = torch.sort(distances)[1]
    selected_indices = sorted_indices[:int((1 - fraction) * len(rankings))]
    return get_voted_ranking_by_sum(rankings[selected_indices])

def get_kemeny_young_ranking(rankings):
    return rk.center(rankings, method='kendalltau')

def get_voting_mechanism(method = "sum"):
    if method == "kemeny_young":
        return get_kemeny_young_ranking
    return aggregate_ranking_with_trim