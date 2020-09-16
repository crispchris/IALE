"""
adapted from https://github.com/JordanAsh/badge/
"""

import torch
import numpy as np
import pdb
from active.strategy import Strategy
from sklearn.neighbors import NearestNeighbors
import pickle
from datetime import datetime
from sklearn.metrics import pairwise_distances
import properties as prop
from torch.utils.data import DataLoader

class CoreSet(Strategy):
    name = 'CoreSetAlt'
    def __init__(self, dataset_pool, valid_dataset, test_dataset, tor=1e-4, device='cuda'):
        super(CoreSet, self).__init__(dataset_pool, [], valid_dataset, test_dataset)
        self.tor = tor
        self.device = device

    def furthest_first(self, X, X_set, n):
        m = np.shape(X)[0]
        if np.shape(X_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = pairwise_distances(X, X_set)
            min_dist = np.amin(dist_ctr, axis=1)

        idxs = []

        for i in range(n):
            idx = min_dist.argmax()
            idxs.append(idx)
            dist_new_ctr = pairwise_distances(X, X[[idx], :])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

        return idxs

    def get_model_pool_embeddings(self, model, device, pool_dataset):
        model.eval()
        pool_dataloader = DataLoader(pool_dataset, batch_size=prop.VAL_BATCH, shuffle=False, num_workers=0)
        embeddings = []
        with torch.no_grad():
            for i, data in enumerate(pool_dataloader):
                inputs, labels = data[0].float(), data[1].long()
                inputs, labels = inputs.to(device), labels.to(device)
                classifications, embedding = model(inputs)
                embeddings.append(embedding)
        return torch.cat(embeddings)

    #def query(self, n):
    def query(self, n, model, train_dataset, pool_dataset):
        t_start = datetime.now()
        lab_embedding = model.get_embeddings(train_dataset[:][0].cuda())
        unlab_embedding = self.get_model_pool_embeddings(model, self.device, pool_dataset)
        #unlab_embedding = model.get_embeddings(pool_dataset[:prop.CORE_SET_SAMPLE_LIMIT][0].cuda())
        lab_embedding = lab_embedding.cpu().numpy()
        unlab_embedding = unlab_embedding.cpu().numpy()
        chosen = self.furthest_first(unlab_embedding, lab_embedding, n)
        not_chosen = []
        for i in range(0, len(pool_dataset)):
            if i not in chosen:
                not_chosen.append(i)
        return chosen, not_chosen


    def query_old(self, n):
        lb_flag = self.idxs_lb.copy()
        embedding = self.get_embedding(self.X, self.Y)
        embedding = embedding.numpy()

        print('calculate distance matrix')
        t_start = datetime.now()
        dist_mat = np.matmul(embedding, embedding.transpose())
        sq = np.array(dist_mat.diagonal()).reshape(len(self.X), 1)
        dist_mat *= -2
        dist_mat += sq
        dist_mat += sq.transpose()
        dist_mat = np.sqrt(dist_mat)
        print(datetime.now() - t_start)
        print('calculate greedy solution')
        t_start = datetime.now()
        mat = dist_mat[~lb_flag, :][:, lb_flag]

        for i in range(n):
            if i % 10 == 0:
                print('greedy solution {}/{}'.format(i, n))
            mat_min = mat.min(axis=1)
            q_idx_ = mat_min.argmax()
            q_idx = np.arange(self.n_pool)[~lb_flag][q_idx_]
            lb_flag[q_idx] = True
            mat = np.delete(mat, q_idx_, 0)
            mat = np.append(mat, dist_mat[~lb_flag, q_idx][:, None], axis=1)

        print(datetime.now() - t_start)
        opt = mat.min(axis=1).max()

        bound_u = opt
        bound_l = opt/2.0
        delta = opt

        xx, yy = np.where(dist_mat <= opt)
        dd = dist_mat[xx, yy]

        lb_flag_ = self.idxs_lb.copy()
        subset = np.where(lb_flag_==True)[0].tolist()

        SEED = 5
        sols = None

        if sols is None:
            q_idxs = lb_flag
        else:
            lb_flag_[sols] = True
            q_idxs = lb_flag_
        print('sum q_idxs = {}'.format(q_idxs.sum()))

        return np.arange(self.n_pool)[(self.idxs_lb ^ q_idxs)]
