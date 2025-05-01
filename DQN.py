import torch
import torch.nn as nn
from Big2 import Big2

class DQN(nn.module):
    """Creates the neural network used for the Deep Q learning Agent"""
    def __init__(self, num_input, num_hidden=100, num_output= 28):
        super(DQN, self).__init__()
        self.fc1 =  nn.Linear(num_input, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_output)
        self.act_fn = nn.ReLU()
        self.act_fn2  = nn.Softmax()

    def forward(self, X):
        X = self.act_fn(self.fc1(X))
        X = self.act_fn(self.fc2(X))
        return X
    




        






