
# coding: utf-8

# #### LSTM For EEG Data
# - First, import everything we're gonna be using

# In[1]:
print('test test test')
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
torch.manual_seed(1)
print('test test test')
# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
# get_ipython().magic(u'load_ext autoreload')
# get_ipython().magic(u'autoreload 2')


# In[2]:

class EEGLSTM(nn.Module):
    
    def __init__(self, seq_len, input_dim, hidden_dim, output_dim = 4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.bidirectional = True #TODO turn this into an arg
        # components needed for our model
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, bidirectional = self.bidirectional)
        # the result if bidirectional is 2x shape if not bidirectional, so account for that
        self.linear = nn.Linear(self.hidden_dim * (2 if self.bidirectional else 1), self.output_dim)
        self.hidden = self.init_hidden(self.hidden_dim)
        
        # hidden layer init
    def init_hidden(self, hidden_dim):
        """Initialize the hidden state in self.hidden
        Dimensions are num_layers * minibatch_size * hidden_dim
        IMPORTANT: Re-initialize this when you want the RNN to forget data (such as training on a new series of timesteps)
        Don't re-init when it's on the same series (because we want to build up the hidden state)
        """
        # num_layers * num_directions, batch, hidden_dim
        bidirectional_mult = 2 if self.bidirectional else 1
        # hidden state and cell state
        return (autograd.Variable(torch.zeros(1 * bidirectional_mult, 1, hidden_dim)),
                autograd.Variable(torch.zeros(1 * bidirectional_mult, 1, hidden_dim)))
    
    def forward(self, input):
        """forwards input through the model"""
        # convert the input into something Pytorch can understand
        input = autograd.Variable(torch.FloatTensor(input)).contiguous()
        # LSTM expects 3-D input: dim of input is expected to be seq_len, batch size, input size
        input = input.view(self.seq_len, 1, -1) # present the sequence seq_len timesteps at a time
        lstm_out, self.hidden = self.lstm(input, self.hidden)
        scores = self.linear(lstm_out.view(self.seq_len,-1))
        return scores
        
        
    


# In[3]:
print('initializing model')
model = EEGLSTM(1, 22, 20) # seq len, input dim, hidden dim
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# In[5]:

# load the data
from load_data import EEGDataLoader
data_loader = EEGDataLoader()
print('loading data')
X_train, y_train, X_test, y_test = data_loader.load_all_data()


# In[ ]:

def train_seq():
    for i in range(X_train.shape[0]):
        for j in range(X_train.shape[1]):
            label = y_train[i][j]
            label = autograd.Variable(torch.LongTensor([int(label % 769)] * 1000))
            sample = X_train[i][j]
            if np.any(np.isnan(sample)):
                print("skipping sample with NaN")
                continue
            assert not np.any(np.isnan(sample))
            model.zero_grad()
            scores = model(sample)
            loss = loss_function(scores, label)
            loss.backward(retain_graph = True)
            optimizer.step()
             # clip the gradient to prevent exploding gradients (probably not needed)
            nn.utils.clip_grad_norm(model.parameters(), 0.99)
            print(loss.data[0])
        break
def train_one_by_one():
    i, j = 0, 0
    iter_count = 0
    for i in range(X_train.shape[0]):
        print("training on dataset: {}".format(i))
        for j in range(X_train.shape[1]):
            print("training on trial {}".format(j))
            sample = X_train[i][j].T
            label = y_train[i][j]
            # hack: deal with NaN
            if np.any(np.isnan(sample)):
                print("skipping sample with NaN")
                continue
            assert not np.any(np.isnan(sample))
            label = autograd.Variable(torch.LongTensor([int(label % 769)]))
#            model.hidden = model.init_hidden(20)
            accumulated_loss = 0
            # present the seq across 1k timesteps, building up the state one at a time
            for k in range(sample.shape[0]):
                model.zero_grad()
                input = sample[0]
                assert input.shape[0] == 22
                scores = model(input)
                loss = loss_function(scores, label)
                accumulated_loss +=loss.data[0]
                out = loss.backward(retain_graph = True)
                optimizer.step()
                # clip the gradient to prevent exploding gradients (probably not needed)
                nn.utils.clip_grad_norm(model.parameters(), 0.99)
                if k % 50 == 0: print("loss at timestep {}: {}".format(k, loss.data[0]))
            lastpred = np.argmax(scores.data.numpy().reshape(4))
            print("last predicted: {}, actual: {}".format(lastpred, y_train[i][j] % 769))
            print("average loss after 1k timesteps: {}".format(accumulated_loss/1000))


train_one_by_one()
#train_seq()


# In[37]:

# prediction code
cur_X, cur_y = X_train[0], y_train[0]
preds = []
for i in range(cur_X.shape[0]):
    sample, label = cur_X[i].T, cur_y[i]
    # present the sequence one at a time, getting a prediction at each timestep
    # hack: deal with NaN
    if np.any(np.isnan(sample)):
        print("skipping sample with NaNs")
        continue
    assert not np.any(np.isnan(sample))
    model.init_hidden(20)
    for i in range(sample.shape[0]):
        input = sample[i]
        scores = model(input)
        nans_exist = np.any(np.isnan(scores.data))
        assert not nans_exist, "nan scores exist"
        predicted_label = np.argmax(scores.data)
    print("last timestep prediction: {}".format(predicted_label + 769))
    preds.append(predicted_label + 769)
diffs = [1 if pred != label else 0 for pred, label in enumerate(preds, list(cur_y))]
assert len(diffs) == 238
errs = sum(diffs)
print(errs/len(diffs))


# In[ ]:



