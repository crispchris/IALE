import numpy as np
import torch
from torch.utils.data import DataLoader
from active.acq_metrics import variation_ratios
from active.strategy import Strategy
import properties as prop
from data.data_helpers import make_tensordataset


class MCDropoutSampling(Strategy):
    name = 'mc-dropout'

    def __init__(self, dataset_pool, valid_dataset, test_dataset, device='cuda:0'):
        super(MCDropoutSampling, self).__init__(dataset_pool, [], valid_dataset, test_dataset)

    def query(self, n, model, train_dataset, pool_dataset):
        device = model.state_dict()['softmax.bias'].device

        predictions = get_mc_pool_preds(model, device, pool_dataset)
        scores = variation_ratios(predictions)

        ordered_ind = torch.argsort(-scores)
        sel_ind = ordered_ind[:n]

        remaining_ind = ordered_ind[n:]

        return sel_ind.cpu().numpy(), remaining_ind.cpu().numpy()


def get_model_pool_preds(model, device, pool_dataset):
    pool_dataloader = DataLoader(pool_dataset, batch_size=prop.VAL_BATCH, shuffle=False, num_workers=0)
    predictions = []
    with torch.no_grad():
        for i, data in enumerate(pool_dataloader):
            inputs, labels = data[0].float(), data[1].long()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions.append(outputs)
    return torch.cat(predictions)


def get_mc_pool_preds(model, device, pool_dataset):
    prediction_list = []
    model.mode_training = True
    model.eval()
    for _ in range(prop.NUM_MC_SAMPLES):
        prediction_list.append(get_model_pool_preds(model, device, pool_dataset))

    model.mode_training = False
    model.train()
    return torch.stack(prediction_list)
