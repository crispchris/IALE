from tqdm import trange
import properties as prop
from data.data_helpers import make_tensordataset, stratified_split_dataset, concat_datasets
from models.CNN import CNN
from models.MLP import mlpMod as MLP
import models.resnet
from models.model_helpers import weights_init
from train_helper import train_validate_model, reinit_seed
from active.policy_helpers import get_state_action

if prop.DATASET.lower() == "mnist":
    from data.mnist import get_policy_training_splits
elif prop.DATASET.lower() == "fmnist":
    from data.fmnist import get_policy_training_splits
elif prop.DATASET.lower() == "kmnist":
    from data.kmnist import get_policy_training_splits
elif prop.DATASET.lower() == "cifar10":
    from data.cifar10 import get_policy_training_splits
elif prop.DATASET.lower() == "cifar100":
    from data.cifar100 import get_policy_training_splits
elif prop.DATASET.lower() == "svhn":
    from data.svhn import get_policy_training_splits
elif prop.DATASET.lower() == "emnist":
    from data.emnist import get_policy_training_splits
#from data.mnist import get_policy_training_splits
from copy import deepcopy as deepcopy
import torch
import numpy as np


def expert(acq_num, model, init_weights, strategies, train_dataset, pool_subset, valid_dataset, test_dataset, device):
    strategy_queries = []
    strategy_acc = []
    trained_weights = deepcopy(model.state_dict())
    for strategy in strategies:
        sel_ind, remain_ind = strategy.query(prop.ACQ_SIZE, model, train_dataset, pool_subset)
        sel_dataset = make_tensordataset(pool_subset, sel_ind)
        curr_train_dataset = concat_datasets(train_dataset, sel_dataset)

        model.load_state_dict(init_weights)
        test_acc = train_validate_model(model, device, curr_train_dataset, valid_dataset, test_dataset)
        model.load_state_dict(trained_weights)

        strategy_acc.append(test_acc)
        strategy_queries.append(sel_ind)

    sel_strategy = strategy_acc.index(max(strategy_acc))
    print("Expert for {} acquisition is {} sampling with model acccuracy {}".format(acq_num, strategies[sel_strategy].name, strategy_acc))
    return strategy_queries[sel_strategy], strategy_acc, strategy_queries


def run_episode(strategies, policy, beta, device, num_worker):
    states, actions = [], []
    # all strategies use same initial training data and model weights
    reinit_seed(prop.RANDOM_SEED)
    if prop.MODEL == "MLP":
        model = MLP().apply(weights_init).to(device)
    if prop.MODEL == "CNN":
        model = CNN().apply(weights_init).to(device)
    if prop.MODEL == "RESNET18":
        model = models.resnet.ResNet18().to(device)
    init_weights = deepcopy(model.state_dict())

    # re-init seed was here before
    use_learner = True if np.random.rand(1) > beta else False
    if use_learner:
        policy = policy.to(device)  # load policy only when learner is used for states

    dataset_pool, valid_dataset, test_dataset = get_policy_training_splits()

    train_dataset, pool_dataset = stratified_split_dataset(dataset_pool, prop.INIT_SIZE, prop.NUM_CLASSES)

    # Initial sampling
    if prop.SINGLE_HEAD:
        my_strategies = []
        for StrategyClass in strategies:
            my_strategies.append(StrategyClass(dataset_pool, valid_dataset, test_dataset))
    if prop.CLUSTER_EXPERT_HEAD:
        UncertaintyStrategieClasses, DiversityStrategieClasses = strategies
        un_strategies = []
        di_strategies = []
        for StrategyClass in UncertaintyStrategieClasses:
            un_strategies.append(StrategyClass(dataset_pool, valid_dataset, test_dataset))
        for StrategyClass in DiversityStrategieClasses:
            di_strategies.append(StrategyClass(dataset_pool, valid_dataset, test_dataset))
    if prop.CLUSTERING_AUX_LOSS_HEAD:
        my_strategies = []
        for StrategyClass in strategies:
            my_strategies.append(StrategyClass(dataset_pool, valid_dataset, test_dataset))


    init_acc = train_validate_model(model, device, train_dataset, valid_dataset, test_dataset)

    t = trange(1, prop.NUM_ACQS + 1, desc="Aquisitions (size {})".format(prop.ACQ_SIZE), leave=True)
    for acq_num in t:
        subset_ind = np.random.choice(a=len(pool_dataset), size=prop.K, replace=False)
        pool_subset = make_tensordataset(pool_dataset, subset_ind)
        if prop.CLUSTER_EXPERT_HEAD:
            un_sel_ind = expert(acq_num, model, init_weights, un_strategies, train_dataset, pool_subset, valid_dataset,
                             test_dataset, device)
            di_sel_ind = expert(acq_num, model, init_weights, un_strategies, train_dataset, pool_subset, valid_dataset,
                                test_dataset, device)
            state, action = get_state_action(model, train_dataset, pool_subset, un_sel_ind=un_sel_ind, di_sel_ind=di_sel_ind)
        if prop.SINGLE_HEAD:
            sel_ind = expert(acq_num, model, init_weights, my_strategies, train_dataset, pool_subset, valid_dataset,
                             test_dataset, device)
            state, action = get_state_action(model, train_dataset, pool_subset, sel_ind=sel_ind)
        if prop.CLUSTERING_AUX_LOSS_HEAD:
            sel_ind = expert(acq_num, model, init_weights, my_strategies, train_dataset, pool_subset, valid_dataset,
                             test_dataset, device)
            state, action = get_state_action(model, train_dataset, pool_subset, sel_ind=sel_ind, clustering=None)
            # not implemented

        states.append(state)
        actions.append(action)
        if use_learner:
            with torch.no_grad():
                if prop.SINGLE_HEAD:
                    policy_outputs = policy(state.to(device)).flatten()
                    sel_ind = torch.topk(policy_outputs, prop.ACQ_SIZE)[1].cpu().numpy()
                if prop.CLUSTER_EXPERT_HEAD:
                    policy_output_uncertainty, policy_output_diversity = policy(state.to(device))
                    # clustering_space = policy_output_diversity.reshape(prop.K, prop.POLICY_OUTPUT_SIZE)
                    # one topk for uncertainty, one topk for diversity
                    diversity_selection = torch.topk(policy_output_diversity.reshape(prop.K), int(prop.ACQ_SIZE/2.0))[1].cpu().numpy()
                    uncertainty_selection = torch.topk(policy_output_uncertainty.reshape(prop.K), int(prop.ACQ_SIZE/2.0))[1].cpu().numpy()
                    sel_ind = (uncertainty_selection, diversity_selection)
                if prop.CLUSTERING_AUX_LOSS_HEAD:
                    # not implemented
                    policy_outputs = policy(state.to(device)).flatten()
                    sel_ind = torch.topk(policy_outputs, prop.ACQ_SIZE)[1].cpu().numpy()

        if prop.SINGLE_HEAD:
            q_idxs = subset_ind[sel_ind]  # from subset to full pool
        if prop.CLUSTER_EXPERT_HEAD:
            unified_sel_ind = np.concatenate((sel_ind[0], sel_ind[1]))
            q_idxs = subset_ind[unified_sel_ind]  # from subset to full pool
        remaining_ind = list(set(np.arange(len(pool_dataset))) - set(q_idxs))

        sel_dataset = make_tensordataset(pool_dataset, q_idxs)
        train_dataset = concat_datasets(train_dataset, sel_dataset)
        pool_dataset = make_tensordataset(pool_dataset, remaining_ind)

        test_acc = train_validate_model(model, device, train_dataset, valid_dataset, test_dataset)

    return states, actions
