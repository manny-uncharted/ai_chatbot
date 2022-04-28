import json
import numpy as np
from nltk_utils import tokenize, stem, bag_of_words

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNetwork

with open('intents.json', 'r') as f:
    intents = json.load(f)

"""Creating an empty list to store all the words"""
all_words = []
tags = [] # an empty list of all patterns
xy = [] # an empty list to later hold the tags and corresponding patterns
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '.', '!', ':', ';', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words] # remove all the words that are not needed, like ?, ., !, :, ;, ,
all_words = sorted(set(all_words)) # remove all the duplicates
tags = sorted(set(tags)) # remove all the duplicates
# print(all_words)
# print(tags)


X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label) # CrossEntropyLoss requires the labels to be categorical

X_train = np.array(X_train)
y_train = np.array(y_train)


"""Defining our hyperparameters"""
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    #dataset[idx]
    """get item function"""
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    """get length function"""
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
# creating dataloader
train_loader = DataLoader(dataset=dataset, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=0)

"""Defining the model"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # if GPU is available, use GPU
model = NeuralNetwork(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)

        # Backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
print ('Final Loss: {:.4f}'.format(loss.item()))

# Saving the model
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'Training complete. Model saved to {FILE}')