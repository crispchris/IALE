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

def active_learn(exp_num, StrategyClass, subsample):
    # all strategies use same initial training data and model weights
    reinit_seed(prop.RANDOM_SEED)
    test_acc_list = []
    model = CNN().apply(weights_init).to(device)
    init_weights = copy.deepcopy(model.state_dict())

    reinit_seed(exp_num*10)
    dataset_pool, valid_dataset, test_dataset = get_data_splits()
    train_dataset, pool_dataset = stratified_split_dataset(dataset_pool, 20, 10)

    # initial data
    strategy = StrategyClass(dataset_pool, valid_dataset, test_dataset, device)

    t = trange(1, prop.NUM_ACQS + 1, desc="Aquisitions (size {})".format(prop.ACQ_SIZE), leave=True)
    for acq_num in t:
        model.load_state_dict(init_weights)

        test_acc = train_validate_model(model, device, train_dataset, valid_dataset, test_dataset)
        test_acc_list.append(test_acc)

        if subsample:
            subset_ind = np.random.choice(a=len(pool_dataset), size=prop.K, replace=False)
            pool_subset = make_tensordataset(pool_dataset, subset_ind)
            sel_ind, remain_ind = strategy.query(prop.ACQ_SIZE, model, train_dataset, pool_subset)
            q_idxs = subset_ind[sel_ind]  # from subset to full pool
            remaining_ind = list(set(np.arange(len(pool_dataset))) - set(q_idxs))
            sel_dataset = make_tensordataset(pool_dataset, q_idxs)
            train_dataset = concat_datasets(train_dataset, sel_dataset)
            pool_dataset = make_tensordataset(pool_dataset, remaining_ind)
        else:
            # all strategies work on k-sized windows in semi-batch setting
            sel_ind, remaining_ind = strategy.query(prop.ACQ_SIZE, model, train_dataset, pool_dataset)
            sel_dataset = make_tensordataset(pool_dataset, sel_ind)
            pool_dataset = make_tensordataset(pool_dataset, remaining_ind)
            train_dataset = concat_datasets(train_dataset, sel_dataset)

        logging.info("Accuracy for {} sampling and {} acquisition is {}".format(strategy.name, acq_num, test_acc))
    return test_acc_list


def get_mean_std(exp_list):
    exp_list = np.array(exp_list)
    mean = np.mean(exp_list, axis=0)
    std = np.std(exp_list, axis=0)
    return mean, std


if __name__ == '__main__':
    torch.cuda.cudnn_enabled = False

    reinit_seed(prop.RANDOM_SEED)

    logging.info("later dumping to {}".format(prop.RESULTS_FILE))
    strategies = [PolicyLearner] 
    results = read_results()
    for strategy in strategies:
        test_acc = [active_learn(exp_num, strategy, subsample=True) for exp_num in range(prop.NUM_EXPS)]
        mean, std = get_mean_std(test_acc)
        results[strategy.name] = [mean.tolist(), std.tolist()]

    """strategies = [RandomSampling, EnsembleSampling, MCDropoutSampling, CoreSetSampling] 
    for strategy in strategies:
        test_acc = active_learn(42, strategy, subsample=False)
        #test_acc = [active_learn(exp_num+42, strategy, subsample=False) for exp_num in range(prop.NUM_EXPS)]
        mean, std = get_mean_std(test_acc)
        results[strategy.name] = [mean.tolist(), std.tolist()]"""

    logging.info("dumping results to {}".format(prop.RESULTS_FILE))
    set_results(results, results_file=prop.RESULTS_FILE)
