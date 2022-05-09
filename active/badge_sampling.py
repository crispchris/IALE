"""
adapted from https://github.com/JordanAsh/badge/
"""

import numpy as np
from torch.utils.data import DataLoader
from active.strategy import Strategy
import pickle
from scipy.spatial.distance import cosine
import sys
import gc
from scipy.linalg import det
from scipy.linalg import pinv as inv
from copy import copy as copy
from copy import deepcopy as deepcopy
import torch
from torch import nn
import torchfile
from torch.autograd import Variable
import torch.optim as optim
from torch.nn import functional as F
import argparse
import torch.nn as nn
from collections import OrderedDict
from scipy import stats
import time
import numpy as np
import scipy.sparse as sp
from itertools import product
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.utils.extmath import row_norms, squared_norm, stable_cumsum
from sklearn.utils.sparsefuncs_fast import assign_rows_csr
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.utils.validation import _num_samples
from sklearn.utils import check_array
from sklearn.utils import gen_batches
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.metrics.pairwise import rbf_kernel as rbf
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances

# kmeans ++ initialization
def init_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    #print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        #print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    gram = np.matmul(X[indsAll], X[indsAll].T)
    val, _ = np.linalg.eig(gram)
    val = np.abs(val)
    vgt = val[val > 1e-2]
    return indsAll

class BadgeSampling(Strategy):
    name = 'badge'
    def __init__(self, dataset_pool, valid_dataset, test_dataset, device='cuda'):
        super(BadgeSampling, self).__init__(dataset_pool, [], valid_dataset, test_dataset)

    def query(self, n, model, train_dataset, pool_dataset):
        gradEmbedding = self.get_grad_embedding(model, pool_dataset).numpy()
        chosen = init_centers(gradEmbedding, n)
        not_chosen = []
        for i in range(0, len(pool_dataset)):
            if i not in chosen:
                not_chosen.append(i)
        return chosen, not_chosen

    # gradient embedding (assumes cross-entropy loss)
    def get_grad_embedding(self, model, pool_dataset):
        embDim = model.get_embedding_dim()
        model.eval()
        # hardcode 10 classes for F/K/MNIST
        nLab = 10 #len(np.unique(Y))
        embedding = np.zeros([len(pool_dataset), embDim * nLab])

        #dataloader = torch.utils.data.DataLoader(pool_dataset, batch_size=len(pool_dataset), shuffle=False)
        dataloader = torch.utils.data.DataLoader(pool_dataset, batch_size=1000, shuffle=False)
        with torch.no_grad():
            for idxs, data in enumerate(dataloader):
                x = data[0].float()
                y = data[1].long()
                if torch.cuda.is_available():
                    x, y = Variable(x.cuda()), Variable(y.cuda())
                else:
                    x, y = Variable(x), Variable(y)
                cout, out = model(x)
                out = out.data.cpu().numpy()
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs, 1)
                for j in range(len(y)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embedding[j][embDim * c: embDim * (c + 1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                        else:
                            embedding[j][embDim * c: embDim * (c + 1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
            return torch.Tensor(embedding)