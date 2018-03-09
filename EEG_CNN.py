import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch import autograd
from load_data import EEGDataLoader
import sys
import torch.nn.functional as F
import numpy as np
# Hyper Parameters
num_epochs = 5
batch_size = 100
learning_rate = 0.001

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 96, kernel_size=3, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=3, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Sequential(
            nn.Linear(33792, 1024),
            nn.ReLU()
            )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 800),
            nn.ReLU()
            )
        self.fc3 = nn.Sequential(
            nn.Linear(800, 256),
            nn.ReLU()
            )
        self.fc4 = nn.Sequential(
            nn.Linear(256, 4)
            )
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return out
        
cnn = CNN()
use_gpu = len(sys.argv) > 1
if use_gpu:
    cnn.cuda()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

data_loader = EEGDataLoader('project_datasets/')
print('loading data')
X_train, y_train, X_test, y_test = data_loader.load_all_data()

# overfit small dataset
X_train, y_train = X_train[0], y_train[0] # 238 * 22 * 1000 or something
for epoch in range(10):
    # 10 epochs, 10 images
    for i in range(10):
        print(i)
        image, label = X_train[i], y_train[i]
        if np.any(np.isnan(image)):
            print('skipping a nan entry')
            continue
        if use_gpu:
            image = autograd.Variable(torch.cuda.FloatTensor(image.reshape((1, 1, image.shape[0], image.shape[1]))))
            label = autograd.Variable(torch.cuda.LongTensor([int(label %769)]))
        else:
            image = autograd.Variable(torch.FloatTensor(image.reshape((1, 1, image.shape[0], image.shape[1]))))
            label = autograd.Variable(torch.LongTensor([int(label %769)]))
        optimizer.zero_grad()
        scores = cnn(image)
        print(scores.shape)
        print(scores.data)
        print(label.shape)
        loss = criterion(scores, label)
        if i % 20 == 0:
            print(i)
            print(loss.data[0])
        loss.backward()
        optimizer.step()

# for i in range(X_train.shape[0]):
#     print(i)
#     X_trial, y_trial = X_train[i], y_train[i]
#     for j in range(X_trial.shape[0]):
#         image, label = X_trial[j], y_trial[j]
#         if np.any(np.isnan(image)):
#             print('skipping a nan entry')
#             continue
#         assert not np.any(np.isnan(image))
#         if len(sys.argv) > 1:
#             image = autograd.Variable(torch.cuda.FloatTensor(image.reshape((1, 1, image.shape[0], image.shape[1]))))
#             label = autograd.Variable(torch.cuda.LongTensor([int(label %769)]))
#         else:
#             image = autograd.Variable(torch.FloatTensor(image.reshape((1, 1, image.shape[0], image.shape[1]))))
#             label = autograd.Variable(torch.LongTensor([int(label %769)]))
#         optimizer.zero_grad()
#         scores = cnn(image)
#         loss = criterion(scores, label)
#         if j % 20 == 0:
#             print(j)
#             print(loss.data[0])
#         loss.backward()
#         optimizer.step()
