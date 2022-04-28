import torch
import torch.nn as nn

"""Creating a feedforward neural network with 
2 hidden layers
and 1 fully connected layer"""
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        # creating an activation function for the neural network
        self.relu = nn.ReLU()

    # Implementing the forward pass
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # No activation and no softmax
        return out