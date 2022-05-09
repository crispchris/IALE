"""
adapted from https://github.com/JordanAsh/badge/
"""

import numpy as np
from active.strategy import Strategy
import pdb
import torch
from torch.utils.data import DataLoader
import properties as prop
from torch.autograd import Variable
from torch.nn import functional as F

class LeastConfidence(Strategy):
    name = "Conf"

    def __init__(self, dataset_pool, valid_dataset, test_dataset, device='cuda'):
        super(LeastConfidence, self).__init__(dataset_pool, [], valid_dataset, test_dataset)
        self.device = device

    def query(self, n, model, train_dataset, pool_dataset):
        idxs = range(0, len(pool_dataset))
        probs = self.predict_prob(model, self.device, pool_dataset)
        U = probs.max(1)[0]
        chosen = U.sort()[1].numpy()[:n]
        not_chosen = []
        for i in range(0, len(pool_dataset)):
            if i not in chosen:
                not_chosen.append(i)
        return chosen, not_chosen


    def predict_prob(self, model, device, pool_dataset):
        dataloader = DataLoader(pool_dataset, batch_size=prop.VAL_BATCH, shuffle=False, num_workers=0)
        model.eval()
        Y = pool_dataset[:][1]
        probs = []
        with torch.no_grad():
            for x, y in dataloader:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                out, emb = model(x)
                prob = F.softmax(out, dim=1)
                probs.append(prob.cpu().data)
        return torch.cat(probs)