import os

import numpy as np
import torch
import heapq
from torch.utils.data import DataLoader, SequentialSampler, ConcatDataset

from active.strategy import Strategy
from active.mc_dropout import MCDropoutSampling
from active.ensemble import EnsembleSampling
from active.coreset import CoreSetSampling
from active.kmeans import KMeansSampling
from active.random import RandomSampling
from data.data_helpers import make_tensordataset
from train_helper import train_validate_model
import properties as prop


class LearnerMLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LearnerMLP, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(hidden_size, output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, sample_embedding, dataset_embedding):
        x = torch.cat((sample_embedding, dataset_embedding), 0)
        y = self.linear1(x)
        y = self.relu1(y)
        y = self.linear2(y)
        y = self.relu2(y)
        y = self.linear3(y)
        y = self.sigmoid(y)
        return y


def monte_carlo_selection(CurrentStrategy, model, mc_length):
    target_strategy = CurrentStrategy()
    select_list = []
    for i in range(mc_length):
        if len(select_list) == 0:
            select_list = target_strategy.query(prop.ACQ_SIZE, model)
        else:
            select_list = np.concatenate((select_list, target_strategy.query(n, model, options)), axis=None)

    unique_elements, count_elements = np.unique(np.array(select_list), return_counts=True)
    zipped_sorted = sorted(zip(unique_elements, count_elements), key=lambda t: t[1], reverse=True)
    best_selections = sorted(list(list(zip(*zipped_sorted))[0][:n]))

    return best_selections


class LearnedSampling(Strategy):
    def __init__(self, dataset_pool, idxs_lb):
        super(LearnedSampling, self).__init__(dataset_pool, idxs_lb)
        input_size = 28*28 + 28*28
        hidden_size = int(input_size / 2)
        output_size = 1

        # Initialize the AL learner
        self.learner = LearnerMLP(input_size, hidden_size, output_size)
        self.optim = torch.optim.Adam(params=self.learner.parameters(), lr=0.001)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optim, gamma=0.5, step_size=20)
        self.learner.to("cuda")

        # Threshold for binary classification
        self.threshold = torch.autograd.Variable(torch.Tensor([0.5]).to("cuda"))
        self.criterion = torch.nn.BCELoss()

    def query(self, n, model, train_dataset, pool_dataset, mode="train"):
        device = model.state_dict()['softmax.bias'].device

        # TRAIN
        # 1. Query one (or all) AL strategy with Monte Carlo for best sample selection -> target
        if mode == "train":
            # Select best stratgey
            strategies = [CoreSetSampling, EnsembleSampling, MCDropoutSampling, RandomSampling]
            highest_acc = 0.
            for StrategyClass in strategies:
                acc, sel_ind = self.run_strategy(StrategyClass, model, train_dataset, pool_dataset, device)
                if acc >= highest_acc:
                    highest_acc = acc
                    final_sel_ind = sel_ind
                    CurrentStrategy = StrategyClass

            # Do Monte Carlo on selected strategy
            # self.monte_carlo_selection(CurrentStrategy, 20)
            target = [1 if i in final_sel_ind else 0 for i in range(0, len(pool_dataset))]

        # 2. Average unlabelled pool
        loader = torch.data.DataLoader(pool_dataset, batch_size=10, num_workers=0, shuffle=False)

        pool_average = 0.
        for images, _ in loader:
            batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
            images = images.view(batch_samples, images.size(1), -1)
            pool_average += images.mean(2).sum(0)

        pool_average /= len(loader.dataset)

        # 3. Run individual sample embeddings with dataset embedding through MLP
        selections = []
        if mode == "train":
            self.learner.train()
        else:
            self.learner.load_state_dict(torch.load(prop.LAL_PATH))

        print("Predicting sample selection with MLP on {} samples".format(len(pool_dataset)))
        for i in range(len(pool_dataset)):
            if mode == "train":
                self.optim.zero_grad()
            # Forward pass
            y = self.learner.forward(torch.Tensor(pool_dataset[i]).to(device), pool_average)
            out = (y > self.threshold).float()

            if mode == "train":
                # Calculate Loss
                loss = self.criterion(out, torch.Tensor([target[i]]).to(device))
                loss = torch.autograd.Variable(loss, requires_grad=True)
                loss.backward()
                self.optim.step()

            # Add sample selection to whole list
            selections.append(y.data.cpu().numpy())

        if train_learner:
            self.lr_scheduler.step()
            torch.save(self.learner.state_dict(), prop.LAL_PATH)

        return selections, remains

    def run_strategy(self, StrategyClass, model, train_dataset, pool_dataset, device):
        strategy = StrategyClass(pool_dataset, self.valid_dataset, self.test_dataset, device)
        sel_ind, remain_ind = strategy.query(prop.ACQ_SIZE, model, train_dataset, pool_dataset)
        sel_dataset = make_tensordataset(pool_dataset, sel_ind)
        train_dataset = ConcatDataset([train_dataset, sel_dataset])
        test_acc = train_validate_model(model, device, train_dataset, self.valid_dataset, self.test_dataset)
        return test_acc, sel_ind


def get_prototypes(target, classes, embeddings):
    def supp_idxs(c):
        return np.where(target == c)

    support_idxs = list(map(supp_idxs, classes.numpy()))
    prototypes = torch.stack([torch.tensor(embeddings[idx_list].mean(0)) for idx_list in support_idxs])
    return prototypes


def get_embeddings(model, loader, device):
    embeddings = None
    with torch.no_grad():
        for batch in iter(loader):
            x, y, lengths = batch
            embedding = model(x.to(device), lengths)
            embedding = embedding.cpu().numpy()
            if embeddings is None:
                embeddings = embedding
            else:
                embeddings = np.concatenate((embeddings, embedding), axis=0)

    return embeddings
