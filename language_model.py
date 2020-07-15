from io import open
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

# read data
def readFromFile(filename):
    text = open(filename, encoding='utf-8').read().strip()
    return text

# preprocessing
def readLinesFromData(directory, seq_len):
    letters = ''
    lines = []
    counter = 0
    for filename in os.listdir(directory):
        counter += 1
        text = readFromFile(os.path.join(directory, filename))
        letters = ''.join(set(letters + text))
        lines.extend([text[i:i + seq_len] for i in range(0, len(text), seq_len)])
        if counter >= 20: # to test network with smaller sample
            break
    return letters, lines

seq_len = 75
directory = os.path.join(os.getcwd(), 'maildir/allen-p/_sent_mail')
letters, lines = readLinesFromData(directory, seq_len)
#letters = string.ascii_letters
#lines = ['hey how are you', 'good im fine']
num_of_letters = len(letters) + 2

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
    return batches


def splitData(data, batch_size):
    training_size = int(len(data) * 4/5 * 4/5)
    validation_size = int(len(data) * 4/5 * 1/5)
    test_size = int(len(data) * 1/5)

    training_data = data[:training_size]
    validation_data = data[training_size: training_size + validation_size]
    test_data = data[training_size + validation_size: training_size + validation_size + test_size]
    
    training_batches = getBatches(batch_size, training_data, seq_len)
    validation_batches = getBatches(batch_size, validation_data, seq_len)
    test_batches = getBatches(batch_size, test_data, seq_len)
    
    return training_batches, validation_batches, test_batches

batch_size = 32

training_batches, validation_batches, test_batches = splitData(lines, batch_size)
#training_batches = getBatches(batch_size, lines, seq_len)
print("Finished preprocessing....")

is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

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

# hyperparameters
num_of_epochs = 10
learning_rate = 0.001
hidden_dim = 100
model = Model(input_size=num_of_letters, output_size=num_of_letters, hidden_dim=hidden_dim, num_of_layers=2)
model.to(device)
softmax = nn.functional.softmax

def plot_graph(all_losses, validation_losses):
    plt.figure()
    plt.plot(all_losses)
    plt.plot(validation_losses)
    plt.show()

def getValidationAccuracyAndLoss(model, criterion, validation_batches):
    model.eval()
    with torch.no_grad():
        validation_loss = []
        matches, total = 0, 0
        for input_seq, target_seq in validation_batches:
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

# loss function
criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
def train(model, training_batches, validation_batches):
    all_losses = []
    validation_losses = []
    for epoch in range(1, num_of_epochs + 1):
        for input_seq, target_seq in training_batches:
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
            validation_accuracy, validation_loss = getValidationAccuracyAndLoss(model, criterion, validation_batches)
            validation_losses.append(validation_loss)
            all_losses.append(total_loss)
            print('Epoch: {}/{}.............'.format(epoch, num_of_epochs), end=' ')
            print("Loss: {:.4f}".format(total_loss), end=' ')
            print("Val Loss: {:.4f}".format(validation_loss), end=' ')
            print("Accuracy: {:.2f}".format(validation_accuracy * 100) + "%", end=' ')
            print("Perplexity: {:.4f}".format(torch.exp(total_loss)))
    plot_graph(all_losses, validation_losses)

train(model, training_batches, validation_batches)
# save model
checkpoint = {'model': model, 'state_dict': model.state_dict(), 'optimiser': optimiser.state_dict}
torch.save(model.state_dict(), os.path.join(os.getcwd(), 'finished_training_checkpoint.pt'))

# predict

# Takes in the model and character as arguments 
# and returns the next character prediction and hidden state
def predict(model, characters):
    print(characters) # previous characters that next character is dependent on
    characters = createInputTensor(characters, len(characters))
    characters.to(device)
    
    output, hidden = model(characters)
    last_output = output[-1][-1]
    prob = nn.functional.softmax(last_output, dim=0).data # can stop here to get probabilities
    print(prob)

    # Take the character with the highest probability score from the output
    char_ind = torch.max(prob, dim=0)[1].item()
    if char_ind == num_of_letters - 1 or char_ind == 0:
        return '', hidden
    return letters[char_ind - 1], hidden

# Takes the desired output length and input characters as arguments, 
# returning the produced string
def sample(model, output_length, start_chars='Message'):
    model.eval() # eval mode
    chars = [ch for ch in start_chars]
    remaining_length = output_length - len(chars)
    for ii in range(remaining_length):
        char, h = predict(model, chars)
        chars.append(char)

    return ''.join(chars)

print(sample(model, 150))