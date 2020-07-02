import copy
import logging
import numpy as np
import torch
from tqdm import trange
import properties as prop
from data.data_helpers import make_tensordataset, stratified_split_dataset, concat_datasets
from models.CNN import CNN
from models.model_helpers import weights_init
from train_helper import train_validate_model, reinit_seed
from active.policy_helpers import get_state_action
from data.mnist import get_policy_training_splits


def expert(acq_num, model, init_weights, strategies, train_dataset, pool_subset, valid_dataset, test_dataset, device):
    strategy_queries = []
    strategy_acc = []
    trained_weights = copy.deepcopy(model.state_dict())
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
    #logging.info("Expert for {} acquisition is {} sampling with model acccuracy {}".format(acq_num, strategies[
    #    sel_strategy].name, strategy_acc))
    return strategy_queries[sel_strategy]


def run_episode(StrategieClasses, policy, beta, device, num_worker):

    states, actions = [], []

    reinit_seed(num_worker)  # all strategies use same initial training data and model weights
    model = CNN().apply(weights_init).to(device)
    init_weights = copy.deepcopy(model.state_dict())

    use_learner = True if np.random.rand(1) > beta else False
    if use_learner:
        policy = policy.to(device)  # load policy only when learner is used for states

    dataset_pool, valid_dataset, test_dataset = get_policy_training_splits()

    train_dataset, pool_dataset = stratified_split_dataset(dataset_pool, prop.INIT_SIZE, prop.NUM_CLASSES)

    # Initial sampling
    strategies = []
    for StrategyClass in StrategieClasses:
        strategies.append(StrategyClass(dataset_pool, valid_dataset, test_dataset))

    init_acc = train_validate_model(model, device, train_dataset, valid_dataset, test_dataset)

    t = trange(1, prop.NUM_ACQS + 1, desc="Aquisitions (size {})".format(prop.ACQ_SIZE), leave=True)
    for acq_num in t:
        subset_ind = np.random.choice(a=len(pool_dataset), size=prop.K, replace=False)
        pool_subset = make_tensordataset(pool_dataset, subset_ind)

        sel_ind = expert(acq_num, model, init_weights, strategies, train_dataset, pool_subset, valid_dataset,
                         test_dataset, device)
        state, action = get_state_action(model, train_dataset, pool_subset, sel_ind)

        states.append(state)
        actions.append(action)
        if use_learner:
            with torch.no_grad():
                policy_outputs = policy(state.to(device)).flatten()
                sel_ind = torch.topk(policy_outputs, prop.ACQ_SIZE)[1].cpu().numpy()

        q_idxs = subset_ind[sel_ind]  # from subset to full pool
        remaining_ind = list(set(np.arange(len(pool_dataset))) - set(q_idxs))

        sel_dataset = make_tensordataset(pool_dataset, q_idxs)
        train_dataset = concat_datasets(train_dataset, sel_dataset)
        pool_dataset = make_tensordataset(pool_dataset, remaining_ind)

        test_acc = train_validate_model(model, device, train_dataset, valid_dataset, test_dataset)

    return torch.cat(states), torch.cat(actions).float()
