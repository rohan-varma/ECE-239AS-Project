import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch import autograd
import sys 
import h5py
import torch.nn.functional as F
import numpy as np

num_epochs = 5
batch_size = 100
learning_rate = 0.001

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1,32, kernel_size=8),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.conv2 = nn.Sequential(
            nn.Conv1d(32,32, kernel_size=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.conv3 = nn.Sequential(
            nn.Conv1d(32,64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.conv4 = nn.Sequential(
            nn.Conv1d(64,64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.conv5 = nn.Sequential(
            nn.Conv1d(64,64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.fc1 = nn.Sequential(
            nn.Linear(1856,1024),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(1024,512),
            nn.ReLU())
        self.fc3 = nn.Sequential(
            nn.Linear(512,4))
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        C, E, T = out.size()
        out = out.view(C, -1)
        #print(out.size())
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


        