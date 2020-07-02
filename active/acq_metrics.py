import torch.nn as nn
import numpy as np
import torch
import properties as prop


def BALD(predictions):
    mc_samples = torch.softmax(predictions, dim=2)

    expected_entropy = - torch.mean(torch.sum(mc_samples * torch.log(mc_samples + 1e-10), dim=-1), dim=0)
    expected_p = torch.mean(mc_samples, dim=0)
    entropy_expected_p = - torch.sum(expected_p * torch.log(expected_p + 1e-10), dim=-1)  # [batch size]
    scores = entropy_expected_p - expected_entropy

    return scores


def variation_ratios(predictions):
    label_predictions = torch.max(predictions, dim=2)[1]
    modes, _ = torch.mode(label_predictions, dim=0)
    num_occurences = (label_predictions == modes).sum(dim=0)
    scores = 1. - num_occurences.float() / label_predictions.shape[0]

    return scores
