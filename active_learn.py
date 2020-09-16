# coding=utf-8
import properties as prop
from active.badge_sampling import BadgeSampling
from active.core_set_alt import CoreSet as CoreSetAlt
from active.least_confidence import LeastConfidence as LeastConfidenceSampling
from active.entropy_sampling import EntropySampling
from active.random import RandomSampling
from active.mc_dropout import MCDropoutSampling
from active.ensemble import EnsembleSampling
from active.policy_learner import PolicyLearner
from train_helper import train_validate_model, reinit_seed
from models.CNN import CNN
from models.MLP import mlpMod as MLP
import models.resnet
import logging
import numpy as np
from models.model_helpers import weights_init
from results_reader import read_results, set_results
import copy
import json
from tqdm import trange
from torch.utils.data import ConcatDataset
import torch
from data.data_helpers import make_tensordataset, stratified_split_dataset, concat_datasets

logging.basicConfig(level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if prop.DATASET.lower() == "mnist":
    from data.mnist import get_data_splits
elif prop.DATASET.lower() == "fmnist":
    from data.fmnist import get_data_splits
elif prop.DATASET.lower() == "kmnist":
    from data.kmnist import get_data_splits
elif prop.DATASET.lower() == "emnist":
    from data.emnist import get_data_splits
elif prop.DATASET.lower() == "cifar10":
    from data.cifar10 import get_data_splits

def active_learn(exp_num, StrategyClass, subsample):
    # all strategies use same initial training data and model weights
    reinit_seed(prop.RANDOM_SEED)
    test_acc_list = []
    if prop.MODEL == "MLP":
        model = MLP().apply(weights_init).to(device)
    if prop.MODEL == "CNN":
        model = CNN().apply(weights_init).to(device)
    if prop.MODEL == "RESNET18":
        model = models.resnet.ResNet18().to(device)
    init_weights = copy.deepcopy(model.state_dict())

    reinit_seed(exp_num*10)
    dataset_pool, valid_dataset, test_dataset = get_data_splits()
    train_dataset, pool_dataset = stratified_split_dataset(dataset_pool, 20, 10)

    # initial data
    strategy = StrategyClass(dataset_pool, valid_dataset, test_dataset, device)

    t = trange(1, prop.NUM_ACQS + 1, desc="Aquisitions (size {})".format(prop.ACQ_SIZE), leave=True)
    for acq_num in t:  # range(1, prop.NUM_ACQS + 1):
        model.load_state_dict(init_weights)  # model.apply(weights_init)

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
    prop.POLICY_FILEPATH = "weights/IALE_CNN_MNIST_FullState_AllExperts_B-1000_init-20_acq-10/policy_99.pth"
    prop.RESULTS_FILE = "../results/temporary_results.json"
    logging.info("later dumping to {}".format(prop.RESULTS_FILE))
    results = read_results()
    print("model name", prop.MODEL)
    print("policy state size", prop.POLICY_INPUT_SIZE)
    print("state", prop.state)
    print("experts", prop.EXPERTS)

    strategies = [PolicyLearner]
    for strategy in strategies:
        test_acc = [active_learn(exp_num, strategy, subsample=True) for exp_num in range(prop.NUM_EXPS)]
        mean, std = get_mean_std(test_acc)
        if args.experts_names != None:
            strategy.name = "IALEv_" + prop.state + "_" + args.experts_names
        results[strategy.name] = [mean.tolist(), std.tolist()]
    strategies = [RandomSampling, BadgeSampling, CoreSetAlt, LeastConfidenceSampling, EntropySampling, EnsembleSampling, MCDropoutSampling]
    for strategy in strategies:
        #test_acc = active_learn(42, strategy, subsample=False)
        test_acc = [active_learn(exp_num+42, strategy, subsample=False) for exp_num in range(prop.NUM_EXPS)]
        mean, std = get_mean_std(test_acc)
        results[strategy.name] = [mean.tolist(), std.tolist()]

    logging.info("dumping results to {}".format(prop.RESULTS_FILE))
    set_results(results, results_file=prop.RESULTS_FILE)
