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
from more_itertools import sort_together
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
from sklearn.externals.six import string_types
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances
import torch
import properties as prop
from torch.utils.data import DataLoader
import numpy as np

def get_state_action(model, train_dataset, pool_subset, un_sel_ind=None, di_sel_ind=None, sel_ind=None, clustering=None):
    if prop.MODEL == "MLP":
        device = 'cuda'
    if prop.MODEL == "CNN":
        device = model.state_dict()['softmax.bias'].device
    if prop.MODEL == "RESNET18":
        device = 'cuda'

    state = get_state(model, device, pool_subset, train_dataset)
    if prop.SINGLE_HEAD:
        action = torch.tensor([1 if ind in sel_ind else 0 for ind in range(state.shape[0])])
        return state, action
    if prop.CLUSTER_EXPERT_HEAD:
        un_action = torch.tensor([1 if ind in un_sel_ind else 0 for ind in range(state.shape[0])])
        di_action = torch.tensor([1 if ind in di_sel_ind else 0 for ind in range(state.shape[0])])
        return state, (un_action, di_action)
    if prop.CLUSTERING_AUX_LOSS_HEAD:
        action = torch.tensor([1 if ind in sel_ind else 0 for ind in range(state.shape[0])])
        clustering = torch.tensor(clustering)
        # FIXME implement
        return state, (action, clustering)



def get_state(model, device, pool_dataset, train_dataset):
    pool_embeddings = get_model_embeddings(model, device, pool_dataset)
    pool_predictions = get_model_predictions(model, device, pool_dataset)
    train_embeddings = get_model_embeddings(model, device, train_dataset)
    train_predictions = get_model_predictions(model, device, train_dataset)
    if prop.ADD_GRADIENT_EMBEDDING:
        gradient_embeddings = get_model_gradient_embeddings(model, device, pool_dataset)
        gradient_embeddings_flat = gradient_embeddings.flatten()
    lab_emb = torch.mean(train_embeddings, dim=0)
    if prop.ADD_POOL_MEAN_EMB:
        pool_emb = torch.mean(pool_embeddings, dim=0)
    train_label_statistics = torch.bincount(train_dataset.tensors[1]).float() / len(train_dataset)
    #  train predictions statistics. if predictions missing, fill up list with 0
    train_pred_lab_unique_cnts = np.unique(train_predictions, return_counts=True)
    for i in range(0, prop.NUM_CLASSES):
        if i not in train_pred_lab_unique_cnts[0]:
            new_tuple = (np.concatenate((train_pred_lab_unique_cnts[0], [i])), np.concatenate((train_pred_lab_unique_cnts[1], [0])))
            train_pred_lab_unique_cnts = new_tuple
    sorted_uniques_cnts = sort_together([train_pred_lab_unique_cnts[0], train_pred_lab_unique_cnts[1]])
    train_pred_label_statistics = torch.Tensor(sorted_uniques_cnts[1] / sum(sorted_uniques_cnts[1]))
    #train_pred_label_statistics = torch.bincount(train_predictions).float() / len(train_predictions)

    state = []
    if prop.ADD_POOL_MEAN_EMB:
        for ind, sample_emb in enumerate(pool_embeddings):
            state.append(torch.cat([lab_emb,
                                    pool_emb,
                                    sample_emb,
                                    train_label_statistics,
                                    train_pred_label_statistics,
                                    get_one_hot(pool_predictions[ind])]))
    if prop.ADD_GRADIENT_EMBEDDING and prop.ADD_PREDICTIONS and not prop.ADD_POOL_MEAN_EMB:
        for ind, sample_emb in enumerate(pool_embeddings):
            state.append(torch.cat([lab_emb,
                                    sample_emb,
                                    train_label_statistics,
                                    train_pred_label_statistics,
                                    gradient_embeddings[ind],
                                    get_one_hot(pool_predictions[ind])]))
    if prop.ADD_GRADIENT_EMBEDDING and not prop.ADD_PREDICTIONS and not prop.ADD_POOL_MEAN_EMB:
        for ind, sample_emb in enumerate(pool_embeddings):
            state.append(torch.cat([lab_emb,
                                    train_label_statistics,
                                    train_pred_label_statistics,
                                    gradient_embeddings[ind]]))
    if prop.ADD_PREDICTIONS and not prop.ADD_GRADIENT_EMBEDDING and not prop.ADD_POOL_MEAN_EMB:
        for ind, sample_emb in enumerate(pool_embeddings):
            state.append(torch.cat([lab_emb,
                                    sample_emb,
                                    train_label_statistics,
                                    train_pred_label_statistics,
                                    get_one_hot(pool_predictions[ind])]))
    for s in state:
        if s.shape[0] == 1565:
            print("state is too small")
    return torch.stack(state)


def get_one_hot(label):
    arr = torch.zeros(prop.NUM_CLASSES)
    arr[label] = 1
    return arr

def get_model_gradient_embeddings(model, device, pool_dataset):
    embDim = model.get_embedding_dim()
    model.eval()
    # hardcode 10 classes for F/K/MNIST
    nLab = 10  # len(np.unique(Y))
    embedding = np.zeros([len(pool_dataset), embDim * nLab])

    dataloader = torch.utils.data.DataLoader(pool_dataset, batch_size=len(pool_dataset), shuffle=False)
    #dataloader = torch.utils.data.DataLoader(pool_dataset, batch_size=10000, shuffle=False)
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

def get_model_embeddings(model, device, dataset):
    model.eval()
    data_loader = DataLoader(dataset, batch_size=prop.VAL_BATCH, shuffle=False, num_workers=0)
    embs = []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs = data[0].float().to(device)
            outputs = model.get_embeddings(inputs)
            embs.append(outputs)
    emb = torch.cat(embs).cpu()
    model.train()
    return emb


def get_model_predictions(model, device, dataset):
    model.eval()
    data_loader = DataLoader(dataset, batch_size=prop.VAL_BATCH, shuffle=False, num_workers=0)
    predictions = []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            outputs, embedding = model(data[0].float().to(device))
            preds = torch.max(outputs, dim=1)[1]
            predictions.append(preds)

    predictions = torch.cat(predictions).cpu()
    model.train()
    return predictions
