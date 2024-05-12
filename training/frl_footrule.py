from args import args
import models

import copy
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import train, Find_rank, FRL_Vote, test

def train_with_frl_footrule(tr_loaders, te_loader):
    print ("#########Federated Learning using Rankings - Footrule Variant############")
    criterion = nn.CrossEntropyLoss().to(args.device)
    model = initialize_model()
    scores = initialize_scores(model)

    epoch = 0
    t_best_acc = 0
    while epoch <= args.FL_global_epochs:
        torch.cuda.empty_cache()
        user_updates = defaultdict(list)

        (round_users, round_malicious, round_benign) = sample_clients()

        run_benign_users(epoch, model, round_benign, user_updates, tr_loaders, criterion)
        if len(round_malicious) > 0:
            run_malicious_users(epoch, model, round_malicious, get_attackers_count(), user_updates, tr_loaders, criterion)

        FRL_Vote(model, user_updates, scores)
        del user_updates
        if epoch % 20 == 0:
            t_loss, t_acc = test(te_loader, model, criterion, args.device) 
            t_best_acc = max(t_best_acc, t_acc)

            log('e %d | malicious users: %d | test acc %.4f test loss %.6f best test_acc %.4f' % (epoch, len(round_malicious), t_acc, t_loss, t_best_acc))
        epoch += 1


def initialize_model():
    args.conv_type = 'MaskConv'
    args.conv_init = 'signed_constant'
    args.bn_type="NonAffineNoStatsBN"    
    
    n_attackers = int(args.nClients * args.at_fractions)
    log("fraction of maliciou clients: %.2f | total number of malicious clients: %d"%(args.at_fractions,n_attackers))
    
    model = getattr(models, args.model)().to(args.device)
    return model

def initialize_scores(model):
    scores={}
    for n, m in model.named_modules():
        if hasattr(m, "scores"):
            scores[str(n)]=m.scores.detach().clone().flatten().sort()[0]

    return scores

def sample_clients():
    n_attackers = get_attackers_count()

    round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
    round_malicious = round_users[round_users < n_attackers]
    round_benign = round_users[round_users >= n_attackers]

    while len(round_malicious)>=args.round_nclients/2:
        round_users = np.random.choice(args.nClients, args.round_nclients, replace=False)
        round_malicious = round_users[round_users < n_attackers]
        round_benign = round_users[round_users >= n_attackers]

    return (round_users, round_malicious, round_benign)

def run_benign_users(epoch, FLmodel, users, user_updates, tr_loaders, criterion):
    for kk in users:
        run_user_training(FLmodel, tr_loaders, criterion, kk, epoch)
        mp = copy.deepcopy(FLmodel)
        optimizer = optim.SGD([p for p in mp.parameters() if p.requires_grad], lr=args.lr*(args.lrdc**epoch), momentum=args.momentum, weight_decay=args.wd)
        
        scheduler = CosineAnnealingLR(optimizer, T_max=args.local_epochs)
        for epoch in range(args.local_epochs):
            train_loss, train_acc = train(tr_loaders[kk], mp, criterion, optimizer, args.device)
            scheduler.step()

        for n, m in mp.named_modules():
                if hasattr(m, "scores"):
                    rank=Find_rank(m.scores.detach().clone())
                    user_updates[str(n)]=rank[None,:] if len(user_updates[str(n)]) == 0 else torch.cat((user_updates[str(n)], rank[None,:]), 0)
                    del rank
        del optimizer, mp, scheduler

def run_malicious_users(epoch, FLmodel, users, n_attackers, user_updates, tr_loaders, criterion):
    sum_args_sorts_mal={}
    for kk in np.random.choice(n_attackers, min(n_attackers, args.rand_mal_clients), replace=False):
        torch.cuda.empty_cache()  
        mp = copy.deepcopy(FLmodel)
        optimizer = optim.SGD([p for p in mp.parameters() if p.requires_grad], lr=args.lr*(args.lrdc**epoch), momentum=args.momentum, weight_decay=args.wd)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.local_epochs)
        for epoch in range(args.local_epochs):
            train_loss, train_acc = train(tr_loaders[kk], mp, criterion, optimizer, args.device)
            scheduler.step()

        for n, m in mp.named_modules():
            if hasattr(m, "scores"):
                rank=Find_rank(m.scores.detach().clone())
                rank_arg=torch.sort(rank)[1]
                if str(n) in sum_args_sorts_mal:
                    sum_args_sorts_mal[str(n)]+=rank_arg
                else:
                    sum_args_sorts_mal[str(n)]=rank_arg
                del rank, rank_arg
        del optimizer, mp, scheduler

    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"):
            rank_mal_agr=torch.sort(sum_args_sorts_mal[str(n)], descending=True)[1]
            for kk in users:
                user_updates[str(n)]=rank_mal_agr[None,:] if len(user_updates[str(n)]) == 0 else torch.cat((user_updates[str(n)], rank_mal_agr[None,:]), 0)
    del sum_args_sorts_mal

def run_user_training(model, tr_loaders, criterion, kk, epoch):
    mp = copy.deepcopy(model)
    optimizer = optim.SGD([p for p in mp.parameters() if p.requires_grad], lr=args.lr*(args.lrdc**epoch), momentum=args.momentum, weight_decay=args.wd)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=args.local_epochs)
    for epoch in range(args.local_epochs):
        train_loss, train_acc = train(tr_loaders[kk], mp, criterion, optimizer, args.device)
        scheduler.step()

    del optimizer, mp, scheduler

def get_attackers_count():
    return int(args.nClients * args.at_fractions)

def log(text):
    print (text)
    with (args.run_base_dir / "output.txt").open("a") as f:
        f.write("\n"+str(text))