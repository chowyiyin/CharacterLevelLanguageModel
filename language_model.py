import io
import torch
import torch.nn as nn
import string
import unicodedata
import time
import math
import random
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import shutil

# read data
def readFromFile(filename):
    try:
      text = open(filename, encoding='utf-8').read().strip()
    except:
      print(filename)
      return '',''
    letters = ''.join(set(text))
    lines = [text[i:i + seq_len] for i in range(0, len(text), seq_len)]
    return letters, lines

# preprocessing
def readLinesFromData(directory, seq_len):
    letters = ''
    lines = []
    for root, directories, filenames in os.walk(directory):
        for directory in directories:
            directory_letters, directory_lines = readLinesFromData(directory, seq_len)
            letters = ''.join(set(letters + directory_letters))
            lines.extend(directory_lines)
            break
        counter = 0
        for filename in filenames:
            counter += 1
            file_letters, file_lines = readFromFile(os.path.join(root, filename))
            letters = ''.join(set(letters + file_letters))
            lines.extend(file_lines)
            if counter >= 10:
                break
    return letters, lines

# prepare input and target tensors
def convertToOneHotEncoding(index_in_letters):
    tensor = torch.zeros(1, num_of_letters)
    tensor[0][index_in_letters] = 1
    return tensor

def createInputTensor(line, max_length):
    tensor = torch.zeros(1, max_length, num_of_letters)
    for letter_index in range(len(line)):
        letter = line[letter_index]
        index_in_letters = letters.find(letter) + 1 #indices shifted by 1 because of padding marker
        tensor[0][letter_index] = convertToOneHotEncoding(index_in_letters)
    for i in range(len(line), max_length):
        tensor[0][i] = convertToOneHotEncoding(0) # add padding
    return tensor

# Target tensor contains indices of letters in the input tensor 
# excluding the first letter and including the EOS
def createTargetTensor(line, max_length):
    letter_indices = [letters.find(line[letter_index]) + 1 for letter_index in range(1, len(line))]
    for i in range(len(line), max_length):
        letter_indices.append(0) # add padding
    letter_indices.append(num_of_letters - 1) # EOS
    return torch.LongTensor(letter_indices)

def getBatches(batch_size, data, seq_len):
    num_of_full_batches = int(float(len(data)) / float(batch_size))
    print(num_of_full_batches)
    data = data[:num_of_full_batches * batch_size]
    batches = []
    counter = 0
    while counter < len(data):
        data_batch = data[counter: counter + batch_size]
        input_tensors = list(map(lambda x: createInputTensor(x, seq_len), data_batch)) 
        target_tensors = list(map(lambda x: createTargetTensor(x, seq_len), data_batch))
        input_seq = torch.cat(input_tensors, dim=0)
        target_seq = torch.stack(target_tensors, dim=0)
        batch = (input_seq, target_seq)
        batches.append(batch)
        counter += batch_size
        yield input_seq, target_seq

def splitData(data, batch_size, segmentation=1000, training_proportion=0.7,
              validation_proportion=0.2, test_proportion=0.1):
    training_data = []
    validation_data = []
    test_data = []
    training_size = int(training_proportion * segmentation)
    validation_size = int(validation_proportion * segmentation)
    test_size = int(test_proportion * segmentation)
    for i in range(0, len(data), segmentation):
      training_end = i + training_size
      training_data.extend(data[i: i + training_end])
      validation_end = training_end + validation_size
      validation_data.extend(data[training_end: validation_end])
      test_end = validation_end + test_size
      test_data.extend(data[validation_end: test_end])
    
    return training_data, validation_data, test_data

# define Model
class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, num_of_layers):
        super(Model, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_of_layers = num_of_layers

        self.input_dropout = nn.Dropout(0.4)
        # Hidden LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_dim, num_of_layers, batch_first = True)
        
        # Fully connected output layer
        self.fc2 = nn.Linear(hidden_dim, output_size)

        
    def forward(self, input):
        batch_size = input.size(0)

        hidden = self.initHidden(batch_size)
        cell_state = self.initCellState(batch_size)

        input = self.input_dropout(input)

        output, (hidden, cell_state) = self.lstm(input, (hidden, cell_state))

        output = self.fc2(output) # output with correct dimensions
        return output, hidden
    
    def initHidden(self, batch_size):
        hidden = torch.zeros(self.num_of_layers, batch_size, self.hidden_dim)
        return hidden
    
    def initCellState(self, batch_size):
        cell_state = torch.zeros(self.num_of_layers, batch_size, self.hidden_dim)
        return cell_state

def save_checkpoint(state, is_best, checkpoint):
   filepath = os.path.join(checkpoint, 'last.pth.tar')
   if not os.path.exists(checkpoint):
       print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
       os.mkdir(checkpoint)
   else:
       print("Checkpoint Directory exists! ")
   torch.save(state, filepath)
   if is_best:
       shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))

def plot_graph(all_losses, validation_losses):
    plt.figure()
    plt.plot(all_losses)
    plt.plot(validation_losses)
    plt.show()

def getValidationAccuracyAndLoss(model, criterion, validation_data, batch_size, seq_len):
    model.eval()
    with torch.no_grad():
        validation_loss = []
        matches, total = 0, 0
        for input_seq, target_seq in getBatches(batch_size, validation_data, seq_len):
            input_seq.to(device)
            output, hidden = model(input_seq)
            criterion_input = output.transpose(1, 2)
            val_loss = criterion(criterion_input, target_seq.long())
            validation_loss.append(val_loss)
            # accuracy
            prob = nn.functional.softmax(output, dim=2).data 
            char_indices = torch.max(prob, dim=2)[1]
            matches += torch.eq(char_indices, target_seq).sum().item()
            total += torch.numel(char_indices)
        validation_accuracy = matches / total
    model.train()
    return validation_accuracy, np.mean(validation_loss)

def printResults(epoch, num_of_epochs, total_loss, validation_loss, 
                 validation_accuracy, perplexity):
  print('Epoch: {}/{}.............'.format(epoch, num_of_epochs), end=' ')
  print("Loss: {:.4f}".format(total_loss), end=' ')
  print("Val Loss: {:.4f}".format(validation_loss), end=' ')
  print("Accuracy: {:.2f}".format(validation_accuracy ) + "%", end=' ')
  print("Perplexity: {:.4f}".format(perplexity))

def createCheckpoint(model, optimiser, letters):
  return {'model': model, 'state_dict': model.state_dict(),
          'optimizer' : optimiser.state_dict(), 'letters': letters}


def train(model, criterion, optimiser, training_data, validation_data, batch_size, seq_len, letters):
    all_losses = []
    validation_losses = []
    highest_accuracy = 0
    for epoch in range(1, num_of_epochs + 1):
      start = time.time()
      for input_seq, target_seq in getBatches(batch_size, training_data, seq_len):
          total_loss = 0
          optimiser.zero_grad()
          input_seq.to(device)
          output, hidden = model(input_seq)
          criterion_input = output.transpose(1, 2)
          loss = criterion(criterion_input, target_seq.long())
          total_loss += loss
          loss.backward()
          optimiser.step()
      if epoch%10 == 0:
          validation_accuracy, validation_loss = getValidationAccuracyAndLoss(
              model, criterion, validation_data,
              batch_size, seq_len)
          # save losses
          validation_losses.append(validation_loss)
          all_losses.append(total_loss)
          checkpoint = createCheckpoint(model, optimiser, letters)
          is_best = validation_accuracy > highest_accuracy
          save_checkpoint(checkpoint, is_best, 'checkpoints')
          if is_best:
            highest_accuracy = validation_accuracy
          
          printResults(epoch, num_of_epochs, total_loss, validation_loss,
                        validation_accuracy * 100, torch.exp(total_loss))
      end = time.time()
      time_for_one_epoch = end - start
      print("epoch: " + str(epoch) + " " + "time: " + str(time_for_one_epoch) + "s")
    plot_graph(all_losses, validation_losses)

def getDevice():
    is_cuda = torch.cuda.is_available()
    device = None
    
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    return device

def main():
    # define lines and letters
    seq_len = 75
    directory = os.path.join(os.getcwd(), 'maildir/allen-p/_sent_mail')
    letters, lines = readLinesFromData(directory, seq_len)
    num_of_letters = len(letters) + 2
    print(len(lines))
    print(num_of_letters)

    # prepare data
    batch_size = 32
    training_data, validation_data, test_data = splitData(lines, batch_size, segmentation=10)

    print("Finished preprocessing....")

    device = getDevice()

    # hyperparameters
    num_of_epochs = 10
    learning_rate = 0.001
    hidden_dim = 50

    # define model
    model = Model(input_size=num_of_letters, output_size=num_of_letters, hidden_dim=hidden_dim, num_of_layers=2)
    model.to(device)
    softmax = nn.functional.softmax

    # loss function
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train(model, criterion, optimiser, training_data, validation_data, batch_size, seq_len)
    
    # save model
    final_checkpoint = createCheckpoint(model, optimiser, letters)
    torch.save(final_checkpoint, os.path.join(os.getcwd(), 'finished_training_checkpoint.pt'))

if __name__ == "__main__":
    main()