import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import properties as prop

def padding_same(kernel_size):
    return math.ceil((kernel_size - 1) / 2)


model_params = {
    'fc1_dropout': 0.25,
    'fc2_dropout': 0.5
}

NUM_CHANNELS = prop.CHANNELS
NUM_CLASSES = prop.NUM_CLASSES


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.embedding_size = 128
        self.mode_training = False
        self.conv1 = nn.Conv2d(NUM_CHANNELS, 32, 4, padding=padding_same(4))
        self.conv2 = nn.Conv2d(32, 32, padding_same(4))
        self.fc1 = nn.Linear(prop.TO_EMBEDDING, self.embedding_size) #self.fc1 = nn.Linear(3872, 128)
        self.softmax = nn.Linear(self.embedding_size, NUM_CLASSES)

    def get_embedding_dim(self):
        return self.embedding_size

    def forward(self, x):
        e1 = self.common_code(x)
        x = self.softmax(e1)
        return x, e1

    def common_code(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.dropout(x, p=model_params['fc1_dropout'], training=self.mode_training)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=model_params['fc2_dropout'], training=self.mode_training)
        return x

    def get_embeddings(self, x):
        with torch.no_grad():
            x = self.common_code(x)
            return x
