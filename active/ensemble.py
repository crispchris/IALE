import torch
from models.CNN import CNN
from models.MLP import mlpMod as MLP
import models.resnet
from models.model_helpers import weights_init
from torch.utils.data import DataLoader
from train_helper import train_validate_model
from active.strategy import Strategy
from active.acq_metrics import variation_ratios
import properties as prop
import logging
from data.data_helpers import make_tensordataset


class EnsembleSampling(Strategy):
    name = 'ensemble'

    def __init__(self, dataset_pool, valid_dataset, test_dataset, device='cuda:0'):
        super(EnsembleSampling, self).__init__(dataset_pool, [], valid_dataset, test_dataset)

    def query(self, n, model, train_dataset, pool_dataset, num_ensembles=5):
        device = 'cuda'
        if prop.MODEL.lower() == "CNN".lower():
            device = model.state_dict()['softmax.bias'].device
        if prop.MODEL.lower() == "RESNET18".lower():
            device = 'cuda'
        if prop.MODEL.lower() == "MLP".lower():
            device = 'cuda'

        predictions = []
        predictions.append(get_model_pool_preds(model, device, pool_dataset))

        ensemble_acc = []
        for i in range(1, num_ensembles):
            if prop.MODEL == "MLP":
                model = MLP().apply(weights_init).to(device)
            if prop.MODEL == "CNN":
                model = CNN().apply(weights_init).to(device)
            if prop.MODEL == "RESNET18":
                model = models.resnet.ResNet18().to(device)
            test_acc = train_validate_model(model, device, train_dataset, self.valid_dataset, self.test_dataset)
            ensemble_acc.append(test_acc)
            predictions.append(get_model_pool_preds(model, device, pool_dataset))

        logging.debug("Ensemble model accuracy is {}".format(ensemble_acc))
        predictions = torch.stack(predictions)
        scores = variation_ratios(predictions)

        ordered_ind = torch.argsort(-scores)
        sel_ind = ordered_ind[:n]

        remaining_ind = ordered_ind[n:]

        return sel_ind.cpu().numpy(), remaining_ind.cpu().numpy()


def get_model_pool_preds(model, device, pool_dataset):
    model.eval()
    pool_dataloader = DataLoader(pool_dataset, batch_size=prop.VAL_BATCH, shuffle=False, num_workers=0)
    predictions = []
    with torch.no_grad():
        for i, data in enumerate(pool_dataloader):
            inputs, labels = data[0].float(), data[1].long()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, embeddings = model(inputs)
            predictions.append(outputs)
    return torch.cat(predictions)
