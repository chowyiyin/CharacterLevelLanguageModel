# -*- coding: utf-8 -*-
"""Model with skorch.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tvou7CafLuUHVQa07aUJR3EsaTmO9RFl

**Defining and training model**
"""

import skorch
import tarfile
import io
import torch
import torch.nn as nn
import string
import unicodedata
import time
import math
import random
import os
import numpy as np
import shutil
import unicodedata
from torch.utils.data import Dataset, DataLoader
from skorch import helper, callbacks
import linecache


separator = '{sep}'

# read data
def readFromFile(input_filename, output_filename, max_length):
    try:
        text = open(input_filename, encoding='utf-8').read().strip()
    except:
        print("Could not open in utf-8: " + input_filename)
        return 0, ''
    letters = ''
    counter = 0
    with open(output_filename, 'a') as output:
        for i in range(0, len(text), seq_len):
            line = text[i: i + seq_len] + separator
            letters = ''.join(set(letters + line))
            counter += 1
            output.write(line)
    return counter, letters

def writeFilenameToFile(test_file, filename):
    with open(test_file, 'a') as output:
        output.write(filename)

# preprocessing
def readLinesFromData(directory, output_filename, max_length, test_file, for_every):
    letters = ''
    num_of_lines = 0
    for root, directories, filenames in os.walk(directory):
        for directory in directories:
            num_from_dir, letters_from_dir = readLinesFromData(directory, output_filename, max_length, test_file, for_every)
            letters = ''.join(set(letters + letters_from_dir))
            num_of_lines += num_from_dir

        for i in range(len(filenames)):
            filename = filenames[i]
            if i % for_every == 0:
                writeFilenameToFile(test_file, os.path.join(root, filename))
            else:
                num_from_file, letters_from_file = readFromFile(os.path.join(root, filename), output_filename, max_length)
                letters = ''.join(set(letters + letters_from_file))
                num_of_lines += num_from_file
    return num_of_lines, letters


class CharacterDataset(Dataset):
    def __init__(self, input_dir, max_length):
        self.input_dir = input_dir
        self.max_length = max_length

        self.processed_file = os.path.join(os.getcwd(), 'enron_processed.txt')
        self.test_file = os.path.join(os.getcwd(), 'enron_test_files.txt')
        if os.path.exists(self.processed_file):
            os.remove(self.processed_file)    
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        for_every = 10
        self.num_of_lines, self.letters = readLinesFromData(input_dir, self.processed_file, max_length, self.test_file, for_every)
        self.num_of_letters = len(self.letters) + 2
        self.split_text = open(self.processed_file, encoding='ascii').read().split(separator)

    # prepare input and target tensors
    def convertToOneHotEncoding(self, index_in_letters):
        tensor = torch.zeros(1, self.num_of_letters)
        tensor[0][index_in_letters] = 1
        return tensor

    def createInputTensor(self, line, max_length):
        input_array = [[0 for i in range(max_length)]]
        for letter_index in range(len(line)):
            try:
                letter = line[letter_index]
                #indices shifted by 1 because of padding marker
                index_in_letters = self.letters.find(letter) + 1
                input_array[0][letter_index] = index_in_letters
            except IndexError:
                print("letter index: " + str(letter_index))
                print("line length: " + str(len(line)))
        for i in range(len(line), max_length):
            input_array[0][i] = 0 # add padding
        return torch.LongTensor(input_array)

    # Target tensor contains indices of letters in the input tensor 
    # excluding the first letter and including the EOS
    def createTargetTensor(self, line, max_length):
        letter_indices = [self.letters.find(line[letter_index]) + 1 for letter_index in range(1, len(line))]
        letter_indices.append(self.num_of_letters - 1) # EOS
        for i in range(len(letter_indices), max_length):
            letter_indices.append(0) # add padding
        return torch.LongTensor(letter_indices)

    def __len__(self):
        return self.num_of_lines

    def __getitem__(self, index):
        line = self.split_text[index]
        input = self.createInputTensor(line, self.max_length)
        target = self.createTargetTensor(line, self.max_length)
        return input.squeeze(), target
    
    def getLetters(self):
        return self.letters, self.num_of_letters
    
# define Model
class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, num_of_layers):
        super(Model, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_of_layers = num_of_layers
        self.embedding = nn.Embedding(input_size, input_size)
        self.input_dropout = nn.Dropout(0.3, inplace=False)
        # Hidden LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_dim, num_of_layers, batch_first = True)
        # Fully connected output layer
        self.fc2 = nn.Linear(hidden_dim, output_size)
        self.norm = nn.LayerNorm(output_size)

    def forward(self, inp, hidden):
        batch_size = inp.size(0)
        inp = self.embedding(inp)
        inp = self.input_dropout(inp)

        output, new_hidden = self.lstm(inp, hidden)

        output_fc = self.fc2(output) # output with correct dimensions
    
        output_norm = self.norm(output_fc)
        
        return (output_norm, new_hidden)
    
    def initState(self, batch_size):
        return (torch.zeros(self.num_of_layers, batch_size, self.hidden_dim),
                torch.zeros(self.num_of_layers, batch_size, self.hidden_dim))

class LSTMNet(skorch.net.NeuralNet):
    def on_epoch_begin(self, *args, **kwargs):
        super().on_epoch_begin(*args, **kwargs)

        self.hidden = self.module_.initState(self.batch_size)

    def train_step(self, X, y):
        self.module_.train()
        
        self.optimizer_.zero_grad()
        
        X = X.to(device)
        y = y.to(device)

        hidden = tuple([each.data for each in self.hidden])
        
        output, self.hidden = self.module_(X, hidden)
        criterion_input = output.transpose(1,2)
        criterion_input.to(device)

        loss = self.get_loss(criterion_input, y)
        loss.to(device)
        loss.backward(retain_graph=True)
        self.optimizer_.step()
        
        return { 'loss': loss, 'y_pred': output }
    
    def validation_step(self, X, y):
        self.module_.eval()
        X = X.to(device)
        y = y.to(device)
        hidden = self.module_.initState(self.batch_size)
        hidden = tuple([each.data for each in hidden])
        output, _ = self.module_(X, hidden)
        criterion_input = output.transpose(1, 2)

        loss = self.get_loss(criterion_input, y)
        
        return {'loss': loss, 'y_pred': output }
    
    def evaluation_step(self, X, **kwargs):
        self.module_.eval()
        X = X.to(device)
        
        hidden = self.module_.initState(self.batch_size)
        hidden = tuple([each.data for each in hidden])
        output, _ = self.module_(X, hidden)

        return output


def getMatchesAndTotal(output, target_seq):
    prob = nn.functional.softmax(torch.from_numpy(output), dim=2) 
    target_seq = torch.stack(target_seq)
    char_indices = torch.max(prob, dim=2)[1]
    target_seq = target_seq[:char_indices.size()[0]]
    matches = torch.eq(char_indices, target_seq).sum().item()
    total = torch.numel(char_indices)
    return matches, total

def ds_accuracy(net, ds, y=None):
    y_true = [y for _, y in ds]
    y_pred = net.predict(ds)
    matches, total = getMatchesAndTotal(y_pred, y_true)
    return (matches / total) * 100

def getLinesAndLetters(seq_len, directory):
    letters, lines = readLinesFromData(directory, seq_len)
    return lines, letters

seq_len = 150

is_cuda = torch.cuda.is_available()
#is_cuda = False
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

def initState(batch_size, num_of_layers, hidden_dim):
    return (torch.zeros(num_of_layers, batch_size, hidden_dim),
            torch.zeros(num_of_layers, batch_size, hidden_dim))

dataset = CharacterDataset('allen-p', seq_len)
letters, num_of_letters = dataset.getLetters()

batch_size = 128
num_of_epochs = 400
learning_rate = 0.001
hidden_dim = int((2/3) * num_of_letters + num_of_letters)
num_of_layers = 2
init_state = initState(batch_size, num_of_layers, hidden_dim)
net = LSTMNet(
    module=Model,
    module__input_size=num_of_letters,
    module__output_size=num_of_letters,
    module__hidden_dim = hidden_dim,
    module__num_of_layers=num_of_layers,
    criterion=torch.nn.CrossEntropyLoss,
    criterion__ignore_index=0,
    optimizer=torch.optim.Adam,
    optimizer__lr=learning_rate,
    batch_size=batch_size,
    max_epochs=num_of_epochs,
    train_split=skorch.dataset.CVSplit(10),
    callbacks=[callbacks.EpochScoring(ds_accuracy, use_caching=False), 
        callbacks.EarlyStopping(patience=30, threshold_mode='abs'), 
        callbacks.Checkpoint(dirname='checkpoints'),
        callbacks.TrainEndCheckpoint(dirname='checkpoints')],
    iterator_train__drop_last=True,
    iterator_valid__drop_last=True
    )


def main():
    with open("letters.txt", 'w') as letters_file:
        letters_file.write(letters)

    print("number_of_letters: " + str(num_of_letters))
 
    print("Finished preprocessing....")

    # hyperparameters
    with torch.autograd.set_detect_anomaly(True):
        net.set_params(device=device)
        net.fit(dataset)

if __name__ == "__main__":
    main()
