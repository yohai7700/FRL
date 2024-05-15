import torch

def vote_sum(rankings):
    sum_args_sorts=torch.sum(rankings, 0)
    return torch.sort(sum_args_sorts)[1]

def trimmed_vote_sum(rankings):
    fraction = 0.2
    ranking = vote_sum(rankings)
    distances = torch.sum(torch.pow(rankings - ranking, 2), 1)
    sorted_indices = torch.sort(distances, descending=True)[1]
    return vote_sum(rankings[sorted_indices[int(fraction * len(rankings)):]])

def trimmed_vote_sum_footrule(rankings):
    fraction = 0.2
    ranking = vote_sum(rankings)
    ranking_indices = torch.sort(ranking)[1]
    distances = torch.Tensor()
    for i in range(len(rankings)):
        current_ranking_indices = torch.sort(rankings[i])[1]
        distances[i] = sum((current_ranking_indices - ranking_indices) ** 2)
    sorted_indices = torch.sort(distances, descending=True)[1]
    return vote_sum(rankings[sorted_indices[int(fraction * len(rankings)):]])