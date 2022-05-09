import torch
import numpy as np
from torch.utils.data import TensorDataset
import properties as prop
from torch.utils.data import DataLoader
from collections import Counter


# Inspired by https://github.com/ej0cl6/deep-active-learning/blob/master/query_strategies/random_sampling.py
class Strategy:
    def __init__(self, dataset_pool, idxs_lb, valid_dataset, test_dataset):
        self.dataset_pool = dataset_pool
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.idxs_lb = idxs_lb
        self.label_stats = []

    def get_label_stats(self):
        labels = self.dataset_pool[self.idxs_lb][1].tolist()
        label_stats = Counter(labels)
        return label_stats

    def query(self, **kwargs):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb
        self.label_stats.append(self.get_label_stats())

    def random_sampling(self, num_selections):
        num_samples = len(self.dataset_pool)
        idxs_tmp = np.arange(num_samples)
        np.random.shuffle(idxs_tmp)
        indices = idxs_tmp[:num_selections]
        return indices

    def stratified_sampling(self, num_selections):
        num_classes = prop.NUM_CLASSES
        assert num_selections % num_classes == 0, "Split size must be a multiple of number of classes"
        split_size = num_selections // num_classes
        data_ind = np.arange(len(self.dataset_pool))
        sel_ind = [np.random.choice(torch.where(self.dataset_pool.tensors[1] == label)[0], split_size) for label in
                   range(num_classes)]
        sel_ind = np.hstack(sel_ind)

        return sel_ind

    def get_embeddings(self, model, device, dataset):
        dataloader = DataLoader(dataset, batch_size=prop.VAL_BATCH, shuffle=False, num_workers=0)
        embeddings = []

        for batch in iter(dataloader):
            X, y = batch
            embedding = model.get_embeddings(X.to(device))
            embeddings.append(embedding.cpu().numpy())

        return np.vstack(embeddings)

    def make_tensordataset(self, data, ind):
        return TensorDataset(data[ind][0], data[ind][1])

    def get_handler(self, handler_type):
        if "pool" in handler_type:
            dataset = self.make_tensordataset(self.dataset_pool, ~self.idxs_lb)
        elif "train" in handler_type:
            dataset = self.make_tensordataset(self.dataset_pool, self.idxs_lb)

        elif "full" in handler_type:
            dataset = self.dataset_pool

        return dataset
