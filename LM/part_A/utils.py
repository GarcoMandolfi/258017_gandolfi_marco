# Corpus adopted: Penn Treebank

import nltk
import os
import torch
import torch.utils.data as data
from functools import partial
from torch.utils.data import DataLoader

global DEVICE

class Vocab:
    def __init__(self, corpus, special_tokens=[]):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v:k for k, v in self.word2id.items()}
    
    def get_vocab(corpus, special_tokens=[]):
        output = {}
        i = 0
        for tk in special_tokens:
            output[tk] = i
            i += 1
        for st in corpus:
            for w in st.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output

class Dataset (data.Dataset):
    def __init__(self, corpus, vocab):
        self.source = []
        self.target = []

        for sent in corpus:
            self.source.append(sent.split()[0:-1])
            self.target.append(sent.split()[1:])
        
        self.source_idx = self.mapping_seq(self.source, vocab)
        self.target_idx = self.mapping_seq(self.target, vocab)
    
    def __len__(self):
        return len(self.source)
    
    def __getitem__(self, idx):
        sample = {
            'source': torch.LongTensor(self.source_idx[idx]),
            'target': torch.LongTensor(self.target_idx[idx])
        }
        return sample        

    def mapping_seq(self, data, vocab):
        res = []
        for seq in data:
            tmp = []
            for w in seq:
                if w in vocab.word2id:
                    tmp.append(vocab.word2id[w])
                else:
                    break
            res.append[tmp]
        return res

def connectToDevice():
    DEVICE = 'cuda:0'

def read_file(path, eos_token="<eos>"):
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output

def retreiveData(path):     # "dataset/PennTreeBank/ptb.valid.txt"
    train_raw = read_file(path)
    dev_raw = read_file(path)
    test_raw = read_file(path)
    return (train_raw, dev_raw, test_raw)

def getDatasets(train_raw, dev_raw, test_raw, vocab):
    train_dataset = Vocab(train_raw, vocab)
    dev_dataset = Vocab(dev_raw, vocab)
    test_dataset = Vocab(test_raw, vocab)
    return (train_dataset, dev_dataset, test_dataset)

def collate(data, pad_token):

    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq
        padded_seqs = padded_seqs.detach()
        return padded_seqs, lengths

    data.sort(key=lambda x: len(x["source"]), reverse=True)
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
    
    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])

    new_item["source"] = source.to(DEVICE)
    new_item["target"] = target.to(DEVICE)
    new_item["number_tokens"] = sum(lengths)
    return new_item

def getDataLoaders(train_dataset, dev_dataset, test_dataset, vocab):
    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=partial(collate, pad_token=vocab.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=128, collate_fn=partial(collate, pad_token=vocab.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=partial(collate, pad_token=vocab.word2id["<pad>"]))