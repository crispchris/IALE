import torch
import properties as prop
from torch.utils.data import DataLoader


def get_state_action(model, train_dataset, pool_subset, sel_ind):
    device = model.state_dict()['softmax.bias'].device

    state = get_state(model, device, pool_subset, train_dataset)
    action = torch.tensor([1 if ind in sel_ind else 0 for ind in range(state.shape[0])])

    return state, action


def get_state(model, device, pool_dataset, train_dataset):
    pool_embeddings = get_model_embeddings(model, device, pool_dataset)
    pool_predictions = get_model_predictions(model, device, pool_dataset)
    train_embeddings = get_model_embeddings(model, device, train_dataset)
    train_predictions = get_model_predictions(model, device, train_dataset)
    lab_emb = torch.mean(train_embeddings, dim=0)
    # TODO FIXME #
    # add pool embedding for ablation study here
    #pool_emb = torch.mean(pool_embeddings, dim=0)
    train_label_statistics = torch.bincount(train_dataset.tensors[1]).float() / len(train_dataset)
    train_pred_label_statistics = torch.bincount(train_predictions).float() / len(train_predictions)

    state = []
    for ind, sample_emb in enumerate(pool_embeddings):
        state.append(torch.cat([lab_emb,
                                    # TODO FIXME #
                                    # add pool embedding for ablation study here
                                    # pool_emb,
                                    sample_emb,
                                    train_label_statistics,
                                    train_pred_label_statistics,
                                    get_one_hot(pool_predictions[ind])]))
    return torch.stack(state)

def get_one_hot(label):
    arr = torch.zeros(prop.NUM_CLASSES)
    arr[label] = 1
    return arr

def get_model_embeddings(model, device, dataset):
    model.eval()
    data_loader = DataLoader(dataset, batch_size=prop.VAL_BATCH, shuffle=False, num_workers=0)
    embs = []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs = data[0].float().to(device)
            outputs = model.get_embeddings(inputs)
            embs.append(outputs)
    emb = torch.cat(embs).cpu()
    model.train()
    return emb


def get_model_predictions(model, device, dataset):
    model.eval()
    data_loader = DataLoader(dataset, batch_size=prop.VAL_BATCH, shuffle=False, num_workers=0)
    predictions = []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            outputs = model(data[0].float().to(device))
            preds = torch.max(outputs, dim=1)[1]
            predictions.append(preds)

    predictions = torch.cat(predictions).cpu()
    model.train()
    return predictions
