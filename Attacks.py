from args import args
from AGRs import *
from eval import *
from misc import *
import torch
import pickle
import torch.nn as nn
import argparse, os, sys, csv, shutil, time, random, operator, pickle, ast, math, copy
import numpy as np
from utils import get_voted_ranking

def frl_greedy_attack(ranking, malicious_count, honest_count):
    honest_ranking = ranking # get_voted_ranking(rankings)
    overall_ranking = honest_ranking
    edge_index_candidates = reversed([edge for edge in range(len(honest_ranking) // 2, len(honest_ranking))])

    target_index = len(overall_ranking) // 2
    for edge_index in edge_index_candidates:
        print(f"checking edge: {edge_index}, len: {len(honest_ranking)}")
        # if not heuristic(overall_ranking, 0, edge_index, honest_ranking, malicious_count, honest_count):
        #     continue
        while target_index != 0:
            print(f"checking target edge: {edge_index}, len: {len(honest_ranking)}")
            candidate_ranking = ranking_move_edge(overall_ranking, target_index, edge_index)
            success = frl_greedy_attack_predict_success(candidate_ranking, honest_ranking, malicious_count, honest_count, honest_ranking[edge_index])

            target_index -= 1
            if success:
                print("success")
                overall_ranking = candidate_ranking
                break
            
        if target_index == 0:
            return overall_ranking
        # for target_index in reversed(range(0, len(overall_ranking) // 2)):
        #     candidate_ranking = ranking_move_edge(overall_ranking, target_index, edge_index)
        #     success = frl_greedy_attack_predict_success(candidate_ranking, honest_ranking, malicious_count, honest_count, honest_ranking[edge_index])
        #     if success:
        #         print("success")
        #         overall_ranking = candidate_ranking
    return overall_ranking

def heuristic(overall_ranking, target_index, edge_index, honest_ranking, malicious_count, honest_count):
    candidate_ranking = ranking_move_edge(overall_ranking, target_index, edge_index)
    return frl_greedy_attack_predict_success(candidate_ranking, honest_ranking, malicious_count, honest_count, honest_ranking[edge_index])

def ranking_move_edge(ranking, target_index, edge_index):
    return torch.cat([ranking[:target_index], ranking[edge_index: edge_index + 1], ranking[target_index: edge_index], ranking[edge_index + 1:]])

def frl_greedy_attack_predict_success(mal_ranking: torch.Tensor, honest_ranking: torch.Tensor, malicious_count, honest_count, edge):
    rankings = torch.cat((mal_ranking.tile((malicious_count,1)), honest_ranking.tile((honest_count, 1))), 0)
    overall_ranking = get_voted_ranking(rankings)
    index = tensor_index_of(overall_ranking, edge)
    return index < len(rankings) / 2

def tensor_index_of(tensor, target_value):
    return (tensor == target_value).nonzero().item()

def our_attack_trmean(all_updates, n_attackers, dev_type='sign', threshold=5.0, threshold_diff=1e-5):
    
    model_re = torch.mean(all_updates, 0)
    
    if dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == 'std':
        deviation = torch.std(all_updates, 0)

    lamda = torch.Tensor([threshold]).cuda()  # compute_lambda_our(all_updates, model_re, n_attackers)

    threshold_diff = threshold_diff
    prev_loss = -1
    lamda_fail = lamda
    lamda_succ = 0

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, all_updates), 0)

        agg_grads = tr_mean(mal_updates, n_attackers)

        loss = torch.norm(agg_grads - model_re)

        if prev_loss < loss:
            # print('successful lamda is ', lamda)
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2
        prev_loss = loss

    mal_update = (model_re - lamda_succ * deviation)
    
    return mal_update



def our_attack_mkrum(all_updates, model_re, n_attackers,dev_type='unit_vec', threshold=5.0, threshold_diff=1e-5):

    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(all_updates, 0)
        
    lamda = torch.Tensor([threshold]).cuda() #compute_lambda_our(all_updates, model_re, n_attackers)
    # print(lamda)
    # threshold_diff = 1e-7
    lamda_fail = lamda
    lamda_succ = 0

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, all_updates), 0)

        agg_grads, krum_candidate = multi_krum(mal_updates, n_attackers, multi_k=True)

        if np.sum(krum_candidate < n_attackers) == n_attackers:
            # print('successful lamda is ', lamda)
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    # print('lambda succ ', lamda_succ)
    mal_update = (model_re - lamda_succ * deviation)
    return mal_update
    mal_updates = torch.stack([mal_update] * n_attackers)
    mal_updates = torch.cat((mal_updates, all_updates), 0)

    return mal_updates