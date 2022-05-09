from torchvision.datasets import EMNIST
from torch.utils.data import TensorDataset
from data.data_helpers import split_dataset, stratified_split_dataset
import properties as prop
import pwd, os
from data.data_helpers import split_dataset, concat_datasets

DATA_PATH = pwd.getpwuid(os.getuid()).pw_dir + '/time_series_data/eMNIST'


def transform_data(data):
    data = data.unsqueeze(1).float().div(255)
    return data


train_dataset = EMNIST(DATA_PATH, split='letters', train=True, download=True) # alternatives: letters, balanced
trainX, trainy = transform_data(train_dataset.data), (train_dataset.targets-1)


train_dataset = TensorDataset(trainX, trainy)


################ test dataset ################################
test_dataset = EMNIST(DATA_PATH, split='letters', train=False, download=True) # alternatives: letters, balanced
testX, testy = transform_data(test_dataset.data), (test_dataset.targets-1)

test_dataset = TensorDataset(testX, testy)
full_dataset = concat_datasets(train_dataset, test_dataset)


def get_data_splits():
    validation_dataset, split_train_dataset = split_dataset(train_dataset, prop.VAL_SIZE)
    return split_train_dataset, validation_dataset, test_dataset


def get_policy_training_splits():
    test_dataset, train_dataset = split_dataset(full_dataset, prop.POLICY_TEST_SIZE)
    validation_dataset, split_train_dataset = split_dataset(train_dataset, prop.VAL_SIZE)
    return split_train_dataset, validation_dataset, test_dataset
