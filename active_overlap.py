# coding=utf-8
from active.random import RandomSampling
from active.mc_dropout import MCDropoutSampling
from active.ensemble import EnsembleSampling
from active.coreset import CoreSetSampling
from active.policy_learner import PolicyLearner
from train_helper import train_validate_model, reinit_seed
from models.CNN import CNN
import logging
import numpy as np
from models.model_helpers import weights_init
import properties as prop
from results.results_reader import read_results, set_results
import copy
import json
from tqdm import trange
from torch.utils.data import ConcatDataset
import torch
from data.data_helpers import make_tensordataset, stratified_split_dataset, concat_datasets

logging.basicConfig(level=logging.INFO)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if prop.DATASET.lower() == "mnist":
    from data.mnist import get_data_splits
elif prop.DATASET.lower() == "fmnist":
    from data.fmnist import get_data_splits
elif prop.DATASET.lower() == "kmnist":
    from data.kmnist import get_data_splits
elif prop.DATASET.lower() == "emnist":
    from data.emnist import get_data_splits


def get_overlap(exp_num, StrategyClasses, subsample):
    # all strategies use same initial training data and model weights
    overlap = {}
    reinit_seed(prop.RANDOM_SEED)
    model = CNN().apply(weights_init).to(device)
    init_weights = copy.deepcopy(model.state_dict())

    reinit_seed(exp_num * 10)
    dataset_pool, valid_dataset, test_dataset = get_data_splits()
    train_dataset, pool_dataset = stratified_split_dataset(dataset_pool, 20, 10)

    # initial data
    base_strategy = PolicyLearner(dataset_pool, valid_dataset, test_dataset, device)

    compare_strategies = [StrategyClass(dataset_pool, valid_dataset, test_dataset, device) for StrategyClass in
                          StrategyClasses]

    t = trange(1, prop.NUM_ACQS + 1, desc="Aquisitions (size {})".format(prop.ACQ_SIZE), leave=True)
    for acq_num in t:  # range(1, prop.NUM_ACQS + 1):
        model.load_state_dict(init_weights)  # model.apply(weights_init)

        test_acc = train_validate_model(model, device, train_dataset, valid_dataset, test_dataset)

        if subsample:
            subset_ind = np.random.choice(a=len(pool_dataset), size=prop.K, replace=False)
            pool_subset = make_tensordataset(pool_dataset, subset_ind)

            sel_ind, remain_ind = base_strategy.query(prop.ACQ_SIZE, model, train_dataset, pool_subset)
            q_idxs = subset_ind[sel_ind]  # from subset to full pool

            # compare with other strategies
            overlap.setdefault('policy', []).append([int(x) for x in sel_ind])

            for strategy in compare_strategies:
                sel_ind, remain_ind = strategy.query(prop.ACQ_SIZE, model, train_dataset, pool_subset)
                overlap.setdefault(strategy.name, []).append([int(x) for x in sel_ind])

            remaining_ind = list(set(np.arange(len(pool_dataset))) - set(q_idxs))
            sel_dataset = make_tensordataset(pool_dataset, q_idxs)
            train_dataset = concat_datasets(train_dataset, sel_dataset)
            pool_dataset = make_tensordataset(pool_dataset, remaining_ind)


        else:
            # all strategies work on k-sized windows in semi-batch setting
            sel_ind, remaining_ind = base_strategy.query(prop.ACQ_SIZE, model, train_dataset, pool_dataset)
            sel_dataset = make_tensordataset(pool_dataset, sel_ind)
            pool_dataset = make_tensordataset(pool_dataset, remaining_ind)
            train_dataset = concat_datasets(train_dataset, sel_dataset)

    return overlap


def concat_dict(dicts):
    sol = {}
    for key in dicts[0].keys():
        for dict in dicts:
            sol.setdefault(key, []).append(dict[key])

    return sol


if __name__ == '__main__':
    torch.cuda.cudnn_enabled = False

    reinit_seed(prop.RANDOM_SEED)

    logging.info("later dumping to {}".format(prop.OVERLAP_RESULTS_FILE))
    strategies = [CoreSetSampling, MCDropoutSampling, EnsembleSampling]
    overlap = [get_overlap(exp_num, strategies, subsample=True) for exp_num in range(prop.NUM_EXPS)]

    overlap = concat_dict(overlap)
    set_results(overlap, results_file=prop.OVERLAP_RESULTS_FILE)
