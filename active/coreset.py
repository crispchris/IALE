import numpy as np
import torch
from torch.utils.data import ConcatDataset
from scipy.spatial.distance import cdist
from active.strategy import Strategy
from data.data_helpers import make_tensordataset


class CoreSetSampling(Strategy):
    name = 'coreset'

    def __init__(self, dataset_pool, valid_dataset, test_dataset, device='cuda:0'):
        super(CoreSetSampling, self).__init__(dataset_pool, [], valid_dataset, test_dataset)
        self.min_distances = None

    def query(self, n, model, train_dataset, pool_dataset, ):
        device = model.state_dict()['softmax.bias'].device

        full_dataset = ConcatDataset([pool_dataset, train_dataset])
        pool_len = len(pool_dataset)

        self.embeddings = self.get_embeddings(model, device, full_dataset)

        idxs_labeled = np.arange(start=pool_len, stop=pool_len + len(train_dataset))

        # Perform kcenter greedy
        self.update_distances(idxs_labeled, idxs_labeled, only_new=False, reset_dist=True)
        sel_ind = []
        for _ in range(n):
            ind = np.argmax(self.min_distances)  # Get sample with highest distance
            assert ind not in idxs_labeled, "Core-set picked index already labeled"
            self.update_distances([ind], idxs_labeled, only_new=True, reset_dist=False)
            sel_ind.append(ind)

        assert len(set(sel_ind)) == len(sel_ind), "Core-set picked duplicate samples"

        remaining_ind = list(set(np.arange(pool_len)) - set(sel_ind))

        return sel_ind, remaining_ind

    def update_distances(self, centers, idxs_labeled, only_new=True, reset_dist=False):
        if reset_dist:
            self.min_distances = None

        if only_new:
            centers = [d for d in centers if d not in idxs_labeled]

        if len(centers) > 0:
            x = self.embeddings[centers]
            dist = cdist(self.embeddings, x, metric='euclidean')

            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1, 1)
            else:
                self.min_distances = np.minimum(self.min_distances, dist)
