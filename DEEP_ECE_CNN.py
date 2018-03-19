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


#PAPER CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1,25, kernel_size=10),
            nn.Conv1d(25,25, kernel_size = 1),
            nn.BatchNorm1d(25),
            nn.ELU(),
            nn.MaxPool1d(3))
        self.conv2 = nn.Sequential(
            nn.Dropout(),
            nn.Conv1d(25,50, kernel_size=10),
            nn.BatchNorm1d(50),
            nn.ELU(),
            nn.MaxPool1d(3))
        self.conv3 = nn.Sequential(
            nn.Dropout(),
            nn.Conv1d(50,100, kernel_size=10),
            nn.BatchNorm1d(100),
            nn.ELU(),
            nn.MaxPool1d(3))
        self.conv4 = nn.Sequential(
            nn.Dropout(),
            nn.Conv1d(100,200, kernel_size=10),
            nn.BatchNorm1d(200),
            nn.ELU(),
            nn.MaxPool1d(3))
        self.fc1 = nn.Sequential(
            nn.Linear(1400,4))
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        C, E, T = out.size()
        out = out.view(C, -1)
        #print(out.size())
        out = self.fc1(out)
        return out
