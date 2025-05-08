# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from utils import *
from model import LM_RNN
import settings
import torch
import torch.optim as optim

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    settings.init()
    if not torch.cuda.is_available():
        print("Failed to connect to gpu")
    else:
        print(torch.cuda.is_available())
        train_raw, dev_raw, test_raw = retreiveData(
            "dataset/PennTreeBank/ptb.train.txt",
            "dataset/PennTreeBank/ptb.valid.txt",
            "dataset/PennTreeBank/ptb.test.txt"
        )

        vocab = Vocab(train_raw, ["<pad>", "<eos>"])

        train_dataset, dev_dataset, test_dataset = getDatasets(
            train_raw, dev_raw, test_raw
        )

        train_loader, dev_loader, test_loader = getDataLoaders(
            train_dataset, dev_dataset, test_dataset, vocab
        )

        output_size = len(vocab.word2id)

        model = LM_RNN(settings.embedding_size,
                       settings.hidden_size,
                       output_size,
                       pad_index=vocab.word2id["<pad>"].to(settings.DEVICE))
        model.apply(init_weights)

        optimizer = optim.SGD(model.parameters(), lr=settings.lr)
        train_criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2id["<pad>"])
        eval_criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2id["<pad>"], reduction='sum')
        print("So far, so good...")
    # Print the results