from torch.utils.data import TensorDataset
import torch
import numpy as np


def make_tensordataset(data, ind):
    return TensorDataset(data[ind][0], data[ind][1])


def concat_datasets(a, b):
    X = torch.cat([a.tensors[0], b.tensors[0]])
    y = torch.cat([a.tensors[1], b.tensors[1]])
    return TensorDataset(X, y)


def split_dataset(dataset, split_size):
    data_ind = np.arange(len(dataset))
    np.random.shuffle(data_ind)
    split_ind, data_ind = data_ind[:split_size], data_ind[split_size:]

    split_dataset, remaining_dataset = make_tensordataset(dataset, split_ind), make_tensordataset(dataset, data_ind)
    return split_dataset, remaining_dataset


def stratified_split_dataset(dataset, split_size, num_classes):
    assert split_size % num_classes == 0, "Split size must be a multiple of number of classes"
    split_size = split_size // num_classes
    data_ind = np.arange(len(dataset))
    sel_ind = [np.random.choice(torch.where(dataset.tensors[1] == label)[0], split_size) for label in
               range(num_classes)]
    sel_ind = np.hstack(sel_ind)
    remain_ind = list(set(data_ind) - set(sel_ind))
    remain_ind.sort()

    split_dataset, remaining_dataset = make_tensordataset(dataset, sel_ind), make_tensordataset(dataset, remain_ind)
    return split_dataset, remaining_dataset
