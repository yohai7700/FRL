# Federated Rank Learning (FRL) - Further Research

This repository contains further research done by us on the Federated Rank Learning (FRL) algorithm, as described in the paper: "Every Vote Counts: Ranking-Based Training of Federated Learning to Resist Poisoning Attacks", this research was done by us as part of a project for the Distributed Computing course in Tel Aviv University.

The following section is based on the correlating section in the original repository's instructions.

## Getting Started

### Prerequisites

Please make sure you have the following dependencies installed:

- Python 3.10.9
- Torch 1.13.1
- Numpy 1.23.5
- Torchvision 0.14.1

### Installation

- I) To download the repository, use the following command:

```bash
git clone https://github.com/yohai7700/FRL.git
```

- II) Create a new conda environment. you can do so using the following command:

```bash
conda create --name FRL_test python=3.10.9
```

- III) Activate the environment:

```bash
conda activate FRL_test
```

- IV) Then, to install the dependencies, run:

```bash
pip install -r requirements.txt
```

## Basic Test

To run a simple experiment on CIFAR10, please run the following command:

```bash
python main.py --data_loc "/CIFAR10/data/" --config experiments/001_config_CIFAR10_Conv8_FRL_1000users_noniid1.0_nomalicious.txt
```

- Note that argument 'data_loc' shows the path to dataset storage (for creation or existing dataset).

This will distribute CIFAR10 over 1000 clients in a non-iid fashion with a Dirichlet distribution parameter $\beta=1.0$. Then, a federated rank learning will be run on top of these 1000 users for 2000 global FL rounds, where 25 clients are chosen for their local update in each round.

### Using Attacks and Variants introduced in this Repository

In order to use the attacks and variants of the FRL algorithm, use the: `FL_type=FrlVariant` argument in the experiment's config.

Note that the algorithm for this variant is implemented in [training/frl_variant.py](training/frl_variant.py).

When using `FrlVariant` the following arguments can be used:

#### Aggregation Method

The `aggregation_method` argument defines how the rankings will be aggregated in the voting phase, the options for the value are listed below:

- `sum` - the default value, aggregates as in the original paper (and algorithm).
- `trimmed_mean_frl` - a variant of the TrimmedMean algorithm for FRL, requires also setting the `ranking_distance_method` argument.

#### Ranking Distance Method

The `ranking_distance_method` argument defines the method which will be used to compute the distance between two rankings, it has the following possible values:

- `l2_norm` - computes the l2-norm of the rankings as if they were vectors.
- `spearman_distance` - uses the Spearman's distance rank correlation measure
- `spearman_footrule` - uses the Spearman's footrule rank correlation measure

#### Ranking Distance Method

The `attack_method` argument defines the method which will be used for how the malicious clients will compute the malicious ranking:

- `reverse` - simply reverses the honest predicted ranking, the same as the original algorithm.
- `swap_half` - swaps between the first and the second half as an implementation of the opportunity-push attack presented in our project.
- `swap_half_reverse` - swaps between the first and the second half and then reverses the last half as an implementation of the opportunity-push attack presented in our project.

## Citation

```
@inproceedings{mozaffarievery,
  title={Every Vote Counts: Ranking-Based Training of Federated Learning to Resist Poisoning Attacks},
  author={Mozaffari, Hamid and Shejwalkar, Virat and Houmansadr, Amir},
  booktitle={32nd USENIX Security Symposium (USENIX Security 23)},
  year={2023}
}
```
