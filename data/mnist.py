from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset
from data.data_helpers import split_dataset, concat_datasets
import properties as prop
import pwd, os

DATA_PATH = pwd.getpwuid(os.getuid()).pw_dir + '/time_series_data/MNIST'

def transform_data(data):
    data = data.unsqueeze(1).float().div(255)
    return data

train_dataset = MNIST(DATA_PATH, train=True, download=True)
trainX, trainy = transform_data(train_dataset.data), train_dataset.targets
train_dataset = TensorDataset(trainX, trainy)

test_dataset = MNIST(DATA_PATH, train=False, download=True)
testX, testy = transform_data(test_dataset.data), test_dataset.targets
test_dataset = TensorDataset(testX, testy)

full_dataset = concat_datasets(train_dataset, test_dataset)

def get_data_splits():
    validation_dataset, split_train_dataset = split_dataset(train_dataset, prop.VAL_SIZE)
    return split_train_dataset, validation_dataset, test_dataset


def get_policy_training_splits():
    test_dataset, train_dataset = split_dataset(full_dataset, prop.POLICY_TEST_SIZE)
    validation_dataset, split_train_dataset = split_dataset(train_dataset, prop.VAL_SIZE)
    return split_train_dataset, validation_dataset, test_dataset
