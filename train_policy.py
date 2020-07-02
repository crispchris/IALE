if __name__ == '__main__':
    import logging
    import torch
    from torch.utils.data import TensorDataset
    from tqdm import trange
    import properties as prop
    from active.random import RandomSampling
    from active.coreset import CoreSetSampling
    from active.ensemble import EnsembleSampling
    from active.mc_dropout import MCDropoutSampling
    from models.Policy import Policy
    from models.model_helpers import weights_init
    from train_helper import train_policy_model
    import pathlib
    from helpers.policy_training_helpers import run_episode
    from train_helper import reinit_seed

    torch.cuda.cudnn_enabled = False

    reinit_seed(prop.RANDOM_SEED)

    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pathlib.Path(prop.POLICY_FOLDER).mkdir(parents=True, exist_ok=True)  # make policy save directories

    # optional training using third expert
    strategies = [EnsembleSampling, MCDropoutSampling]
    # strategies = [EnsembleSampling, MCDropoutSampling, CoreSetSampling]

    policy = Policy().apply(weights_init)
    all_states, all_actions = [], []
    t = trange(0, prop.NUM_EPISODES, desc="Episode".format(prop.ACQ_SIZE), leave=True)
    for episode in t:
        logging.warning("Starting episode {}".format(episode + 1))
 
        beta = 0.5
        # exponential beta
        if prop.BETA == "exp":
            beta = pow(0.9, episode)
        elif prop.BETA == "fixed":
            beta = 0.5
        curr_states, curr_actions = run_episode(strategies,
                                                    policy,
                                                    beta,
                                                    device,
                                                    num_worker=episode)
        all_states.append(curr_states)
        all_actions.append(curr_actions)

        states, actions = torch.cat(all_states), torch.cat(all_actions)

        policy_data = TensorDataset(states, actions.unsqueeze(-1))
        policy.apply(weights_init).to(device)  # train the policy on first gpu
        loss = train_policy_model(policy, device, policy_data)

        logging.warning("Policy loss at the end of episode {} is {}".format(episode + 1, loss))

        torch.save(policy.state_dict(), prop.POLICY_FOLDER + '/policy_{}.pth'.format(episode))
        torch.save(states, prop.POLICY_FOLDER + '/states_{}.pth'.format(episode))
        torch.save(actions, prop.POLICY_FOLDER + '/actions_{}.pth'.format(episode))
