# coding=utf-8
import properties as prop
import logging
logging.basicConfig(level=logging.INFO)
import argparse
# import dependencies
from active.badge_sampling import BadgeSampling
from active.core_set_alt import CoreSet as CoreSetAltSampling
from active.least_confidence import LeastConfidence as LeastConfidenceSampling
from active.entropy_sampling import EntropySampling
from active.random import RandomSampling
from active.mc_dropout import MCDropoutSampling
from active.ensemble import EnsembleSampling
#
# overwrite defaults from properties for easier use upon publication
#


parser = argparse.ArgumentParser(description='Run active learners.')
parser.add_argument('--budget', help="total samples to acquire", default=1000)
parser.add_argument('--acq_size', help="samples acquired per iteration", default=10)
parser.add_argument('--sub_pool', help="random subsampled pool for policy to select from to use", default=2000)
parser.add_argument('--init_size', help="initial labeled samples to use", default=20)
parser.add_argument('--dataset', help="Dataset to use", default="fmnist")
parser.add_argument('--model', help="model to use", default="cnn")
parser.add_argument('--heuristic', help="heuristic to use", default="policy")
parser.add_argument('--policy_path', help="policy weights", default="./weights/IALE_CNN_MNIST_FullState_AllExperts_B-1000_init-20_acq-10/policy_99.pth")
parser.add_argument('--num_experiments', help="random inits repetitions", default=1)
parser.add_argument('--name', help="key to identify results", default="reproduction")
args = parser.parse_args()

acq_size = int(args.acq_size)
sub_pool = int(args.sub_pool)
init_size = int(args.init_size)
dataset = args.dataset
heuristic = args.heuristic
budget = int(args.budget)
namehelper = args.name
prop.MODEL = args.model.lower()
prop.NUM_EXPS = int(args.num_experiments)
prop.POLICY_FILEPATH = args.policy_path

prop.DATASET = dataset
if dataset.lower() == "cifar10" or dataset.lower() == "cifar100" or dataset.lower() == "svhn":
    prop.CHANNELS = 3
    prop.TO_EMBEDDING = 8192
if dataset.lower() == "cifar100":
    prop.NUM_CLASSES = 100
if dataset.lower() == "emnist":
    #prop.NUM_CLASSES = 47
    prop.NUM_CLASSES = 26

prop.ACQ_SIZE = acq_size
prop.NUM_ACQS = int((budget-init_size) / acq_size)
prop.LABELING_BUDGET = prop.NUM_ACQS * prop.ACQ_SIZE
prop.K = sub_pool
prop.INIT_SIZE = init_size

strategies = []
if heuristic.lower() == "random":
    strategies = [RandomSampling]
elif heuristic.lower() == "badge":
    strategies = [BadgeSampling]
elif heuristic.lower() == "coreset":
    strategies = [CoreSetAltSampling]
elif heuristic.lower() == "conf":
    strategies = [LeastConfidenceSampling]
elif heuristic.lower() == "entropy":
    strategies = [EntropySampling]
elif heuristic.lower() == "ensemble":
    strategies = [EnsembleSampling]
elif heuristic.lower() == "mcdropout":
    strategies = [MCDropoutSampling]
logging.info(f"name helper {namehelper}")
logging.info(f"|D_sub| = {prop.K}")
logging.info(f"labeling budget {prop.LABELING_BUDGET}")
logging.info(f"acquisition size {prop.ACQ_SIZE}")
logging.info(f"NUM_ACQS {prop.NUM_ACQS + 1}")
logging.info(f"init size {prop.INIT_SIZE}")
logging.info(f"Heuristic {strategies}")
logging.info(f"dataset {prop.DATASET}")
logging.info(f"model {prop.MODEL}")
logging.info(f"repetitions {prop.NUM_EXPS}")
logging.info(f"policy weights {prop.POLICY_FILEPATH}")

prop.RESULTS_FILE = f"results/dev_{prop.DATASET}-{prop.NUM_CLASSES}-{prop.MODEL}_{namehelper}_{heuristic}_episode-{prop.POLICY_FILEPATH[-6:-4]}_{prop.NUM_EXPS}x-B={prop.LABELING_BUDGET}-ACQ={prop.ACQ_SIZE}-subpool={prop.K}-INIT={prop.INIT_SIZE}-NUM={prop.NUM_ACQS}.json"
logging.info(f"later dumping to {prop.RESULTS_FILE}")

from active.policy_learner import PolicyLearner
from train_helper import train_validate_model, reinit_seed
from models.CNN import CNN
from models.MLP import mlpMod as MLP
import models.resnet
import numpy as np
from models.model_helpers import weights_init
from results_reader import read_results, set_results
import copy
from tqdm import trange
import torch
from data.data_helpers import make_tensordataset, stratified_split_dataset, concat_datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if prop.DATASET.lower() == "mnist":
    from data.mnist import get_data_splits
elif prop.DATASET.lower() == "fmnist":
    from data.fmnist import get_data_splits
    prop.NUM_CLASSES = 10
elif prop.DATASET.lower() == "kmnist":
    from data.kmnist import get_data_splits
elif prop.DATASET.lower() == "cifar10":
    from data.cifar10 import get_data_splits
elif prop.DATASET.lower() == "cifar100":
    from data.cifar100 import get_data_splits
elif prop.DATASET.lower() == "svhn":
    from data.svhn import get_data_splits
elif prop.DATASET.lower() == "emnist":
    from data.emnist import get_data_splits

def active_learn(exp_num, StrategyClass, subsample):
    # all strategies use same initial training data and model weights
    reinit_seed(prop.RANDOM_SEED)
    test_acc_list = []
    if prop.MODEL.lower() == "mlp":
        model = MLP().apply(weights_init).to(device)
    if prop.MODEL.lower() == "cnn":
        model = CNN().apply(weights_init).to(device)
    if prop.MODEL.lower() == "resnet18":
        model = models.resnet.ResNet18().to(device)
    init_weights = copy.deepcopy(model.state_dict())

    reinit_seed(exp_num*10)
    dataset_pool, valid_dataset, test_dataset = get_data_splits()
    train_dataset, pool_dataset = stratified_split_dataset(dataset_pool, 2*prop.NUM_CLASSES, prop.NUM_CLASSES)#


    # initial data
    strategy = StrategyClass(dataset_pool, valid_dataset, test_dataset, device)
    # calculate the overlap of strategy with other strategies
    strategies = [MCDropoutSampling, EnsembleSampling, EntropySampling, LeastConfidenceSampling,
                         CoreSetAltSampling, BadgeSampling]
    overlapping_strategies = []
    for StrategyClass in strategies:
        overlapping_strategies.append(StrategyClass(dataset_pool, valid_dataset, test_dataset))
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
    results = read_results()
    logging.info(f"model name {prop.MODEL}")
    logging.info(f"policy state size { prop.POLICY_INPUT_SIZE}")
    logging.info(f"state { prop.state}")
    logging.info(f"experts { prop.EXPERTS}")
    logging.info(f"num classes { prop.NUM_CLASSES}")
    if heuristic == "policy":
        strategies = [PolicyLearner]
        for strategy in strategies:
            test_acc = [active_learn(exp_num, strategy, subsample=True) for exp_num in range(prop.NUM_EXPS)]#prop.NUM_EXPS)]
            mean, std = get_mean_std(test_acc)
            results[strategy.name] = [mean.tolist(), std.tolist()]
            logging.info("dumping results to {}".format(prop.RESULTS_FILE))
            set_results(results, results_file=prop.RESULTS_FILE)
    else:
        for strategy in strategies:
            test_acc = [active_learn(exp_num+42, strategy, subsample=False) for exp_num in range(prop.NUM_EXPS)]
            mean, std = get_mean_std(test_acc)
            results[strategy.name] = [mean.tolist(), std.tolist()]
            logging.info("dumping results to {}".format(prop.RESULTS_FILE))
            set_results(results, results_file=prop.RESULTS_FILE)
