from torchvision.datasets import CIFAR100
import torchvision
from torch.utils.data import TensorDataset
from data.data_helpers import split_dataset, concat_datasets
import properties as prop
import pwd, os
import torch

DATA_PATH = pwd.getpwuid(os.getuid()).pw_dir + '/time_series_data/CIFAR100'


def transform_data(data):
    data = torch.Tensor(data)
    return data


train_dataset = CIFAR100(DATA_PATH, train=True, download=True)
if torchvision.__version__ == "0.5.0":
    trainX, trainy = transform_data(train_dataset.data), train_dataset.targets
else:
    trainX, trainy = transform_data(train_dataset.train_data), train_dataset.train_labels
trainy = torch.Tensor(trainy).long()
trainX = trainX.transpose(3,1).transpose(2,3)
train_dataset = TensorDataset(trainX, trainy)

test_dataset = CIFAR100(DATA_PATH, train=False, download=True)
if torchvision.__version__ == "0.5.0":
    testX, testy = transform_data(test_dataset.data), test_dataset.targets
else:
    testX, testy = transform_data(test_dataset.test_data), test_dataset.test_data
testy = torch.Tensor(testy).long()
testX = testX.transpose(3,1).transpose(2,3)
test_dataset = TensorDataset(testX, testy)

full_dataset = concat_datasets(train_dataset, test_dataset)


def get_data_splits():
    validation_dataset, split_train_dataset = split_dataset(train_dataset, prop.VAL_SIZE)
    return split_train_dataset, validation_dataset, test_dataset


def get_policy_training_splits():
    test_dataset, train_dataset = split_dataset(full_dataset, prop.POLICY_TEST_SIZE)
    validation_dataset, split_train_dataset = split_dataset(train_dataset, prop.VAL_SIZE)
    return split_train_dataset, validation_dataset, test_dataset
