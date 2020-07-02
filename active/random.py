# coding=utf-8
import numpy as np
from active.strategy import Strategy


class RandomSampling(Strategy):
    name = 'random'

    def __init__(self, dataset_pool, valid_dataset, test_dataset, device='cuda:0'):
        super(RandomSampling, self).__init__(dataset_pool, [], valid_dataset, test_dataset)

    def query(self, n, model, train_dataset, pool_dataset):
        data_ind = np.arange(len(pool_dataset))
        np.random.shuffle(data_ind)
        sel_ind, remaining_ind = data_ind[:n], data_ind[n:]
        return sel_ind, remaining_ind
