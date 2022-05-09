import properties as prop
from active.ensemble import EnsembleSampling
from active.mc_dropout import MCDropoutSampling
from active.core_set_alt import CoreSet as CoreSetAltSampling
from active.badge_sampling import BadgeSampling
from active.entropy_sampling import EntropySampling
from active.least_confidence import LeastConfidence as LeastConfidenceSampling
from active.random import RandomSampling
import logging
import torch
from tqdm import trange
from models.Policy import Policy
from models.model_helpers import weights_init
from train_helper import train_policy_model
import pathlib
from helpers.policy_training_helpers import run_episode
import torch.multiprocessing as mp
from train_helper import reinit_seed

if __name__ == '__main__':
    prop.NUM_CLASSES = 10  # adapt this for other datasets
    torch.cuda.cudnn_enabled = False
    reinit_seed(prop.RANDOM_SEED)
    mp.set_start_method('spawn')
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pathlib.Path(prop.POLICY_FOLDER).mkdir(parents=True, exist_ok=True)  # make policy save directories
    assert(prop.CLUSTER_EXPERT_HEAD + prop.CLUSTERING_AUX_LOSS_HEAD + prop.SINGLE_HEAD == 1)
    oneHeadStrategies = [MCDropoutSampling, EnsembleSampling, EntropySampling, LeastConfidenceSampling, CoreSetAltSampling, BadgeSampling]
    #oneHeadStrategies = [RandomSampling for i in range(0, prop.NUM_OF_RANDOM_EXPERTS)]
    policy = Policy().apply(weights_init)
    all_states, all_actions = [], []
    logging.info(f"Using dataset {prop.DATASET}")
    t = trange(0, prop.NUM_EPISODES, desc="Episode".format(prop.ACQ_SIZE), leave=True)
    for episode in t:
        logging.warning("Starting episode {}, storing to {}".format(episode + 1, prop.POLICY_FOLDER))
        beta = 0.5
        # exponential beta
        if prop.BETA == "exp":
            beta = pow(0.9, episode)
        elif prop.BETA == "fixed":
            beta = 0.5
        # if np.rand < beta, its less likely over time to use policy when using exponential beta
        # if np.rand > beta, its more likeliy over time to use policy.
        strategies = (oneHeadStrategies)
        curr_states, curr_actions = run_episode(strategies,
                                                policy,
                                                beta,
                                                device,
                                                num_worker=episode)
        all_states.append(torch.cat(curr_states))
        all_actions.append(torch.cat(curr_actions))

        # initialize weights for current episode
        policy.apply(weights_init).to(device)
        loss = train_policy_model(policy, device, all_states, all_actions)

        logging.warning("Policy loss at the end of episode {} is {}".format(episode + 1, loss))
        logging.warning(f"Saving model: {prop.PLOT_NAME}/policy_{episode}.pth")

        torch.save(policy.state_dict(), prop.POLICY_FOLDER + '/policy_{}.pth'.format(episode))
        # save states and actions for resumption of interrupted training
        # torch.save(all_states, prop.POLICY_FOLDER + '/states_{}.pth'.format(episode))
        # torch.save(all_actions, prop.POLICY_FOLDER + '/actions_{}.pth'.format(episode))
