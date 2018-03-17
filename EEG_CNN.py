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
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 96, kernel_size=3, padding=2),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.conv4 = nn.Sequential(
            nn.Conv1d(96, 256, kernel_size=3, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.conv6 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=3, padding=2),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.fc1 = nn.Sequential(
            nn.Linear(352768, 1024),
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

X_train, y_train = X_train[0], y_train[0] # 238 * 22 * 1000 or something
print(X_train.shape, y_train.shape)
batch_size = 100
i = 0
for epoch in range(2000):
    while i < X_train.shape[0]:
        # sample a batch
        image_batch, label_batch = X_train[i:i+50], y_train[i:i+50]
        image_batch = image_batch.reshape((image_batch.shape[0], image_batch.shape[1] * image_batch.shape[2]))
        print("batch shape: {} {}".format(image_batch.shape, label_batch.shape))
        assert(not np.any(np.isnan(image_batch)))
        if np.any(np.isnan(image_batch)):
            print('skipping this entire batch lol')
            i+=50
        else:
            if use_gpu:
                image = autograd.Variable(torch.cuda.FloatTensor(image_batch.reshape((image_batch.shape[0], 1, image_batch.shape[1]))))
                label = autograd.Variable(torch.cuda.LongTensor([int(label %769) for label in label_batch]))    
            else:
                image = autograd.Variable(torch.FloatTensor(image_batch.reshape((image_batch.shape[0], 1, image_batch.shape[1]))))
                label = autograd.Variable(torch.LongTensor([int(label %769) for label in label_batch]))
            i+=50
            optimizer.zero_grad()
            scores = cnn(image)
            print(scores.cpu().data.numpy().shape)
            loss = criterion(scores, label)
            if epoch % 1 == 0:
                print(loss.data[0])
                preds = np.argmax(scores.cpu().data.numpy(), axis=1)
                preds = preds + 769
                correct = (preds == np.array(label_batch)).astype(int)
                print("{} correct predictions out of 50".format(np.sum(correct)))
            loss.backward()
            optimizer.step()

    # for i in range(X_train.shape[0]):
    #     image, label = X_train[i], y_train[i]
    #     if np.any(np.isnan(image)):
    #         print('skipping a nan entry')
    #         continue
    #     if use_gpu:
    #         image = autograd.Variable(torch.cuda.FloatTensor(image.reshape((1, 1, image.shape[0], image.shape[1]))))
    #         label = autograd.Variable(torch.cuda.LongTensor([int(label %769)]))
    #     else:
    #         image = autograd.Variable(torch.FloatTensor(image.reshape((1, 1, image.shape[0], image.shape[1]))))
    #         label = autograd.Variable(torch.LongTensor([int(label %769)]))
    #     optimizer.zero_grad()
    #     scores = cnn(image)
    #     loss = criterion(scores, label)
    #     if i % 20 == 0:
    #         print(loss.data[0])
    #         print('predicted: {}, actual: {}'.format(np.argmax(scores.cpu().data.numpy()), y_train[i] % 769))
    #     loss.backward()
    #     optimizer.step()
        #if i == X_train.shape[0]: break # out of the inner loop

# gauge accuracy on the training dataset
predictions = []
labels = []
for i in range(X_train.shape[0]):
    image, label = X_train[i], y_train[i]
    image = image.reshape(1, image.shape[0] * image.shape[1])
    labels.append(label)
    assert not np.any(np.isnan(image))
    if np.any(np.isnan(image)):
        print('skipping a nan entry')
        continue
    if use_gpu:
        image = autograd.Variable(torch.cuda.FloatTensor(image_batch.reshape((1, 1, image_batch.shape[1]))))
        label = autograd.Variable(torch.cuda.LongTensor([int(label %769) for label in label_batch]))    
    else:
        image = autograd.Variable(torch.FloatTensor(image_batch.reshape((1, 1, image_batch.shape[1]))))
        label = autograd.Variable(torch.LongTensor([int(label %769) for label in label_batch]))
    optimizer.zero_grad()
    scores = cnn(image)
    prediction = np.argmax(scores.cpu().data.numpy())
    predictions.append(prediction)
#    if i == X_train.shape[0]: break # out of the inner loop

diffs = [1 if pd != (lb % 769) else 0 for pd, lb in zip(predictions, labels)]
error = sum(diffs)/len(diffs)
print("accuracy: {}".format(1 - error))