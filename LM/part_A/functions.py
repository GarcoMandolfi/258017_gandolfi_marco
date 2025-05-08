import math
import torch
import torch.nn as nn

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

    with torch.no_grad():
        for sample in data:
            output = model(sample['source'])
            loss = criterion(output, sample['target'])
            loss_array.append(loss.item())
            tokens_number.append(sample["number_tokens"])

    resulting_loss = sum(loss_array) / sum(tokens_number)
    ppl = math.exp(resulting_loss)
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