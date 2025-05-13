import math
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import copy

import settings

def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()
    loss_array = []
    tokens_number = []

    for sample in data:
        optimizer.zero_grad()
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        tokens_number.appens(sample["number_tokens"])
        loss.backwards()
        torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        optimizer.step()
    
    return sum(loss_array)/sum(tokens_number)

def eval_loop(data, criterion, model):
    model.eval()
    loss_array = []
    tokens_number = []
    resulting_loss = []

    with torch.no_grad():
        for sample in data:
            output = model(sample['source'])
            loss = criterion(output, sample['target'])
            loss_array.append(loss.item())
            tokens_number.append(sample["number_tokens"])

    ppl = math.exp(sum(loss_array) / sum(tokens_number))
    resulting_loss = sum(loss_array) / sum(tokens_number)
    return ppl, resulting_loss

def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)

def train_model(model, train_loader, dev_loader, test_loader,
                optimizer, train_criterion, eval_criterion):
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    epochs = tqdm(range(1,settings.epochs_number))

    for e in epochs:
        loss = train_loop(train_loader, optimizer, train_criterion, model)
        if e % 1 == 0:
            sampled_epochs.append(e)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, eval_criterion, model)
            losses_dev.append(np.asarray(loss_dev).mean())
            epochs.set_description("PPL: %f" % ppl_dev)
            if ppl_dev < best_ppl:
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                settings.patience = 3
            else:
                settings.patience -= 1
            
            if settings.patience <= 0:
                break
    
    best_model.to(settings.DEVICE)
    final_ppl, _ = eval_loop(test_loader, eval_criterion, model)
    print('Test ppl: ', final_ppl)
    return best_model