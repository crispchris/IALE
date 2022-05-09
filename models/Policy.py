import torch
import torch.nn as nn
import properties as prop


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        input_size = prop.POLICY_INPUT_SIZE
        output_size = 1
        self.common_code = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        if prop.SINGLE_HEAD:
            self.fc = nn.Linear(128, 1)
            self.sigmoid = nn.Sigmoid()
        if prop.CLUSTER_EXPERT_HEAD:
            self.fc_un = nn.Linear(128, 1)
            self.sigmoid_un = nn.Sigmoid()
            self.fc_di = nn.Linear(128, 1)
            self.sigmoid_di = nn.Sigmoid()
        if prop.CLUSTERING_AUX_LOSS_HEAD:
            # not implemented
            self.fc = nn.Linear(128, 1)
            self.sigmoid = nn.Sigmoid()
            self.fc_clustering = nn.Linear(128, 1)
            self.sigmoid_clustering = nn.Sigmoid()

    def forward(self, x):
        embedding = self.common_code(x)
        if prop.SINGLE_HEAD:
            x = self.fc(embedding)
            x = self.sigmoid(x)
            return x
        if prop.CLUSTER_EXPERT_HEAD:
            uncertainty_vector = self.fc_un(embedding)
            uncertainty_vector = self.sigmoid_un(uncertainty_vector)
            diversity_vector = self.fc_di(embedding)
            diversity_vector = self.sigmoid_di(diversity_vector)
            return uncertainty_vector, diversity_vector
        if prop.CLUSTERING_AUX_LOSS_HEAD:
            score = self.fc_un(embedding)
            score = self.sigmoid_un(score)
            # not implemented
            clustering = self.fc_di(embedding)
            clustering = self.sigmoid_di(clustering)
            return score, clustering
