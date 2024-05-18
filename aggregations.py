from args import args
import torch

def get_voted_ranking_by_sum(rankings):
    sum_args_sorts=torch.sum(rankings, 0)
    idxx=torch.sort(sum_args_sorts)[1]
    return idxx

def aggregate_ranking_with_trim(rankings):
    fraction = 0.2
    ranking = get_voted_ranking_by_sum(rankings)
    distances = compute_distances(rankings, ranking)
    sorted_indices = torch.sort(distances)[1]
    selected_indices = sorted_indices[:int((1 - fraction) * len(rankings))]
    return get_voted_ranking_by_sum(rankings[selected_indices])

def compute_distances(elements: torch.Tensor, element: torch.Tensor):
    if args.ranking_distance_method == "spearman_distance":
        return compute_distances_spearman_distance(elements, element)
    if args.ranking_distance_method == "spearman_footrule":
        return compute_distances_spearman_footrule(elements, element)
    if args.ranking_distance_method == "l2_norm":
        return compute_distances_norm_l2(elements, element)
    
    raise MissingArgumentError("ranking_distance_method")

def compute_distances_spearman_distance(elements: torch.Tensor, element: torch.Tensor) -> torch.Tensor:
    distances = list()
    sorted_element = torch.sort(element)[1]
    for i in range(len(elements)):
        sorted_current = torch.sort(elements[i])[1]
        distances.append(torch.sum((sorted_current - sorted_element) ** 2))
    return torch.Tensor(distances)


def compute_distances_spearman_footrule(elements: torch.Tensor, element: torch.Tensor) -> torch.Tensor:
    distances = list()
    sorted_element = torch.sort(element)[1]
    for i in range(len(elements)):
        sorted_current = torch.sort(elements[i])[1]
        distances.append(torch.sum(torch.abs((sorted_current - sorted_element))))
    return torch.Tensor(distances)

def compute_distances_norm_l2(elements, element):
    return ((elements - element) ** 2).sum(axis = 1)


def get_voting_mechanism(method = "sum"):
    if method == "sum":
        return get_voted_ranking_by_sum
    if method == "trimmed_mean_frl":
        return aggregate_ranking_with_trim

class MissingArgumentError(RuntimeError):
    def __init__(self, name):
        RuntimeError.__init__(self, f"Missing Argument: {name}")