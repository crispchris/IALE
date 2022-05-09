import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np
import properties as prop

DIMS = 28*28
NUM_CLASSES = prop.NUM_CLASSES
model_params = {
    'fc1_dropout': 0.25,
    'fc2_dropout': 0.5
}

# mlp model class
class mlpMod(nn.Module):
    def __init__(self, dim=DIMS, embSize=128):
        super(mlpMod, self).__init__()
        self.embSize = embSize
        self.dim = int(np.prod(dim))
        self.lm1 = nn.Linear(self.dim, embSize)
        self.lm2 = nn.Linear(embSize, NUM_CLASSES)
        self.mode_training = False
    def forward(self, x):
        x = x.view(-1, self.dim)
        emb = F.relu(self.lm1(x))
        emb = F.dropout(emb, p=model_params['fc2_dropout'], training=self.mode_training)
        out = self.lm2(emb)
        return out, emb

    def common_code(self, x):
        x = x.view(-1, self.dim)
        emb = F.relu(self.lm1(x))
        emb = F.dropout(emb, p=model_params['fc2_dropout'], training=self.mode_training)
        out = self.lm2(emb)
        return out

    def get_embedding_dim(self):
        return self.embSize
    def get_embeddings(self, x):
        with torch.no_grad():
            x = x.view(-1, self.dim)
            emb = F.relu(self.lm1(x))
            return emb

