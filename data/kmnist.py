from torchvision.datasets import KMNIST
from torch.utils.data import TensorDataset
from data.data_helpers import split_dataset, stratified_split_dataset
import properties as prop
import pwd, os

DATA_PATH = pwd.getpwuid(os.getuid()).pw_dir + '/time_series_data/kMNIST'


def transform_data(data):
    data = data.unsqueeze(1).float().div(255)
    return data


train_dataset = KMNIST(DATA_PATH, train=True, download=True)
trainX, trainy = transform_data(train_dataset.data), train_dataset.targets

#validation_dataset, train_dataset = split_dataset(TensorDataset(trainX, trainy), prop.VAL_SIZE)
#trainX, trainy = train_dataset.tensors[0], train_dataset.tensors[1]

train_dataset = TensorDataset(trainX, trainy)

# train_size = 2000
# train_dataset, _ = split_dataset(train_dataset, train_size)

################ test dataset ################################
test_dataset = KMNIST(DATA_PATH, train=False, download=True)
testX, testy = transform_data(test_dataset.data), test_dataset.targets

test_dataset = TensorDataset(testX, testy)

def get_data_splits():
    validation_dataset, split_train_dataset = split_dataset(train_dataset, prop.VAL_SIZE)
    # train_size = 2000
    # split_train_dataset, _ = split_dataset(split_train_dataset, train_size)
    return split_train_dataset, validation_dataset, test_dataset
