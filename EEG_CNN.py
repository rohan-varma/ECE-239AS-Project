import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch import autograd
from load_data import EEGDataLoader
import sys
import numpy as np
# Hyper Parameters
num_epochs = 5
batch_size = 100
learning_rate = 0.001

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(40000, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
        
cnn = CNN()
if len(sys.argv) > 1:
    cnn.cuda()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

data_loader = EEGDataLoader('project_datasets/')
print('loading data')
X_train, y_train, X_test, y_test = data_loader.load_all_data()

# train the model
for i in range(X_train.shape[0]):
    X_trial, y_trial = X_train[i], y_train[i]
    for j in range(X_trial.shape[0]):
        image, label = X_trial[j], y_trial[j]
        if np.any(np.isnan(image)):
            print('skipping a nan entry')
            continue
        assert not np.any(np.isnan(image))
        print(image.shape)
        image = autograd.Variable(torch.FloatTensor(image.reshape((1, 1, image.shape[0], image.shape[1]))))
        scores = cnn(image)
        print(scores.shape)
        print(label)
