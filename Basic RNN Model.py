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

# read data
def readLinesFromFile(filename):
    lines = open(filename, encoding='utf-8').read().strip()
    lines_array = lines.split("\n")
    ascii_array = []
    for line in lines_array:
        converted_line = convertUnicodeToAscii(line)
        if len(converted_line) != 0:
            ascii_array.append(converted_line)
    return ascii_array

# preprocessing
letters = string.ascii_letters
num_of_letters = len(letters) + 2 # Index 0 for padding marker and last index for EOS marker

def convertUnicodeToAscii(unicode_string):
    separator = ''
    return separator.join(
        char for char in unicodedata.normalize('NFD', unicode_string)
        if unicodedata.category(char) != 'Mn'
        and char in letters
    )

lines = []
directory = r'/Users/chowyiyin/Documents/Character-Level-RNN/maildir/allen-p/_sent_mail'
counter = 0
for filename in os.listdir(directory):
    counter += 1
    print(counter)
    lines.extend(readLinesFromFile(os.path.join(directory, filename)))
    if counter >= 20:
        break
max_length = len(max(lines, key=len))

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

seq_len = max_length - 1
batch_size = len(lines)
input_seq = []
target_seq = []
for i in range(len(lines)):
    line = lines[i]
    input = createInputTensor(line, max_length) 
    target = createTargetTensor(line, max_length)
    input_seq.append(input)
    target_seq.append(target)

input_seq = torch.cat(input_seq, dim=0)
target_seq = torch.stack(target_seq, dim=0)

print("Finished preprocessing....")

is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. 
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

        # Hidden LSTM layer
        #self.rnn = nn.RNN(input_size, hidden_dim, num_of_layers, batch_first = True) # uses the default tanh
        self.lstm = nn.LSTM(input_size, hidden_dim, num_of_layers, batch_first = True)
        # Fully connected output layer
        self.fc2 = nn.Linear(hidden_dim, output_size)
        
    def forward(self, input):
        batch_size = input.size(0)

        hidden = self.initHidden(batch_size)
        cell_state = self.initCellState(batch_size)

        #output, hidden = self.rnn(input, hidden)
        output, (hidden, cell_state) = self.lstm(input, (hidden, cell_state))

        output = output.contiguous().view(-1, self.hidden_dim)

        output = self.fc2(output)

        return output, hidden
    
    def initHidden(self, batch_size):
        hidden = torch.zeros(self.num_of_layers, batch_size, self.hidden_dim)
        return hidden
    
    def initCellState(self, batch_size):
        cell_state = torch.zeros(self.num_of_layers, batch_size, self.hidden_dim)
        return cell_state

# hyperparameters
num_of_epochs = 900
learning_rate = 0.01
hidden_dim = num_of_letters
model = Model(input_size=num_of_letters, output_size=num_of_letters, hidden_dim=num_of_letters, num_of_layers=1)
model.to(device)

# loss function
criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
all_losses = []
for epoch in range(1, num_of_epochs + 1):
    #permutations = torch.randperm(input_seq.size()[0])
    total_loss = 0
    optimiser.zero_grad()
    #indices = permutations[i:i+batch_size]
    #input_batch, target_batch = input_seq[indices], target_seq[indices]
    input_seq.to(device)
    output, hidden = model(input_seq)
    loss = criterion(output, target_seq.view(-1).long())
    total_loss += loss
    loss.backward()
    optimiser.step()
    all_losses.append(total_loss)
    if epoch%10 == 0:
        print('Epoch: {}/{}.............'.format(epoch, num_of_epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))

plt.figure()
plt.plot(all_losses)
plt.show()

# save model
torch.save(model.state_dict(), "/Users/chowyiyin/Documents/model")

# predict
# This function takes in the model and character as arguments 
# and returns the next character prediction and hidden state
def predict(model, characters):
    print(characters)
    # One-hot encoding our input to fit into the model
    characters = createInputTensor(characters, len(characters))
    characters.to(device)
    
    output, hidden = model(characters)
    last_output = output[-1]

    prob = nn.functional.softmax(last_output, dim=0).data
    print(prob)
    # Taking the class with the highest probability score from the output
    char_ind = torch.max(prob, dim=0)[1].item()
    if char_ind == num_of_letters - 1 or char_ind == 0:
        return '', hidden
    return letters[char_ind - 1], hidden

# This function takes the desired output length and input characters as arguments, 
# returning the produced sentence
def sample(model, out_len, start='evans'):
    model.eval() # eval mode
    # First off, run through the starting characters
    chars = [ch for ch in start]
    size = out_len - len(chars)
    # Now pass in the previous characters and get a new one
    for ii in range(size):
        char, h = predict(model, chars)
        chars.append(char)

    return ''.join(chars)

print(sample(model, 100))