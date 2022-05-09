import torch
from torch.utils.data import DataLoader
import numpy as np
import properties as prop
from active.strategy import Strategy
from models.Policy import Policy
from active.policy_helpers import get_state
import logging


class PolicyLearner(Strategy):
    name = 'IALE_' + str(prop.state) + "-" + str(prop.EXPERTS)

    def __init__(self, dataset_pool, idxs_lb, valid_dataset, test_dataset, device='cuda', policy_file=None):
        super(PolicyLearner, self).__init__(dataset_pool, idxs_lb, valid_dataset, test_dataset)

        self.policy = Policy()
        if policy_file is not None:
            logging.info("LOADING THE MODE FROM {}".format(policy_file))
            self.policy.load_state_dict(torch.load(policy_file))
            logging.info("USING THE MODE FROM {}".format(policy_file))
        else:
            logging.info("LOADING THE MODE FROM {}".format(prop.POLICY_FILEPATH))
            self.policy.load_state_dict(torch.load(prop.POLICY_FILEPATH))
            logging.info("USING THE MODE FROM {}".format(prop.POLICY_FILEPATH))
        self.policy.to(device)
        self.policy.eval()

    def query(self, n, model, train_dataset, pool_dataset):
        if prop.MODEL.lower() == "mlp":
            device = 'cuda'
        if prop.MODEL.lower() == "cnn":
            device = model.state_dict()['softmax.bias'].device
        if prop.MODEL.lower() == "resnet18":
            device = 'cuda'

        state = get_state(model, device, pool_dataset, train_dataset)
        state_dataloader = DataLoader(state, batch_size=prop.VAL_BATCH, shuffle=False, num_workers=0)
        if prop.SINGLE_HEAD:
            with torch.no_grad():
                policy_outputs = []
                for i, data in enumerate(state_dataloader):
                    inputs = data.float().to(device)
                    outputs = self.policy(inputs)
                    policy_outputs.append(outputs)
                policy_outputs = torch.cat(policy_outputs).cpu().flatten()
                sel_ind = torch.topk(policy_outputs, n)[1].cpu().numpy()
                remaining_ind = list(set(np.arange(len(pool_dataset))) - set(sel_ind))
                return sel_ind, remaining_ind
        if prop.CLUSTER_EXPERT_HEAD:
            pol_out_un = []
            pol_out_div = []
            with torch.no_grad():
                for i, data in enumerate(state_dataloader):
                    inputs = data.float().to(device)
                    uncertain, diverse = self.policy(inputs)
                    pol_out_un.append(uncertain)
                    pol_out_div.append(diverse)
            pol_out_un = torch.cat(pol_out_un).cpu().flatten()
            pol_out_div = torch.cat(pol_out_div).cpu().flatten()

            sel_ind_0 = torch.topk(pol_out_un, int(n / 2.0))[1].cpu().numpy()
            sel_ind_1 = torch.topk(pol_out_div, int(n / 2.0))[1].cpu().numpy()
            sel_ind = np.concatenate((sel_ind_0, sel_ind_1))
            remaining_ind = list(set(np.arange(len(pool_dataset))) - set(sel_ind))
            return sel_ind, remaining_ind
        if prop.CLUSTERING_AUX_LOSS_HEAD:
            # not implemented
            return sel_ind, remaining_ind



