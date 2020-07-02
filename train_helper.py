import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import torch.optim as optim
import numpy as np
import torch.nn as nn
import logging
import properties as prop
from tqdm import trange


def reinit_seed(seed):
    #print("reinit_seed with {}".format(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def train_ensemble_models(models, devices, train_dataset, valid_dataset, test_dataset, ensemble=True):
    if ensemble == False or len(models) == 1:
        model_acc = train_validate_model(models[0], devices[0], train_dataset, valid_dataset, test_dataset)
        print("Model acc is {}".format(model_acc))

    else:
        # acc_list = workers.starmap(fully_train,
        #                            [(models[i], devices[i], train_dataset, valid_dataset, test_dataset)
        #                             for i in range(len(models))])
        acc_list = []
        for i in range(len(models)):
            acc_list.append(train_validate_model(models[i], devices[i], train_dataset, valid_dataset, test_dataset))

        print(acc_list)

        model_acc = np.mean(np.array(acc_list))

    return model_acc


def train_validate_model(model, device, train_dataset, valid_dataset, test_dataset, valid_setting=False):
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(prop.NUM_EPOCHS):
        train_one_epoch(epoch, model, train_dataset, device, optimizer, criterion)
        valid_acc, valid_loss, report = validate_model(epoch, model, valid_dataset, device, criterion)

    if valid_setting:
        return valid_acc
    else:
        test_acc, test_report = test_model(model, test_dataset, device)
        return test_acc


def train_policy_model(model, device, train_dataset):
    optimizer = optim.Adam(model.parameters())
    class_weights = torch.bincount(train_dataset.tensors[1].flatten().int())
    class_weights = class_weights.float() / torch.sum(class_weights)
    class_weights = class_weights[[-1, 0]]
    criterion = nn.BCELoss(reduction='none')
    # criterion = nn.BCELoss()
    for epoch in range(prop.NUM_EPOCHS_POLICY):
        loss = train_one_epoch(epoch, model, train_dataset, device, optimizer, criterion, class_weights=class_weights)
        logging.debug("{} epoch training loss is {}".format(epoch, loss))

    return loss


def train_one_epoch(epoch, model, train_dataset, device, optimizer, criterion, class_weights=None):
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=prop.TRAIN_BATCH, shuffle=True)
    model.train()
    model.mode_training = True # enable Dropout layers during training
    running_loss = 0.0
    for i, data in enumerate(train_dataloader):
        inputs, labels = data[0].float().to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        if class_weights is not None:
            weight_ = class_weights[labels.view(-1).long()].view_as(labels)
            loss = loss * weight_.to(device)
            loss = loss.sum()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    model.mode_training = False  # disable Dropout layers after training
    running_loss = running_loss / (i + 1)
    logging.debug("{} epoch training loss is {}".format(epoch, running_loss))
    return running_loss


def validate_model(epoch, model, valid_dataset, device, criterion):
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=prop.VAL_BATCH, shuffle=False)
    model.eval()
    ground_truth = []
    predictions = []
    running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(valid_dataloader):
            inputs, labels = data[0].float().to(device), data[1].to(device)
            ground_truth.extend(labels.tolist())
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.tolist())

    running_loss = running_loss / (i + 1)
    valid_acc = accuracy_score(ground_truth, predictions)

    logging.debug("{} epoch validation accuracy and loss is {} and {}".format(epoch, valid_acc, running_loss))
    return valid_acc, running_loss, _


def test_model(model, test_dataset, device):
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=prop.VAL_BATCH, shuffle=False)
    model.eval()

    ground_truth = []
    predictions = []
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            inputs, labels = data[0].float().to(device), data[1].to(device)
            ground_truth.extend(labels.tolist())
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.tolist())

    test_acc = accuracy_score(ground_truth, predictions)  # f1_score(ground_truth, predictions, average='macro')
    test_report = classification_report(ground_truth, predictions)
    test_matrix = confusion_matrix(ground_truth, predictions)

    logging.debug("Test report is {}".format(test_report))
    logging.debug(test_matrix)
    return test_acc, test_report
