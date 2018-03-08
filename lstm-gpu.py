
# coding: utf-8

# #### LSTM For EEG Data
# - First, import everything we're gonna be using

# In[1]:
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
from load_data import EEGDataLoader
torch.manual_seed(1)

class EEGLSTM(nn.Module):
    
    def __init__(self, seq_len, input_dim, hidden_dim, output_dim = 4, batch_size = 1, bidirectional=True, gpu_enabled=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.bidirectional = bidirectional
        self.gpu_enabled = gpu_enabled
        self.batch_size = batch_size
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
        if self.gpu_enabled:
            return (autograd.Variable(torch.zeros(1 * bidirectional_mult, 1, hidden_dim)).cuda(),
                autograd.Variable(torch.zeros(1 * bidirectional_mult, 1, hidden_dim)).cuda())
        else:
            return (autograd.Variable(torch.zeros(1 * bidirectional_mult, 1, hidden_dim)),
                autograd.Variable(torch.zeros(1 * bidirectional_mult, 1, hidden_dim)))
    
    def forward(self, input):
        """forwards input through the model"""
        # convert the input into something Pytorch can understand
        if self.gpu_enabled:
            input = autograd.Variable(torch.FloatTensor(input)).contiguous().cuda()
        else:
            input = autograd.Variable(torch.FloatTensor(input)).contiguous()
        # LSTM expects 3-D input: dim of input is expected to be seq_len, batch size, input size
        input = input.view(self.seq_len, 1, -1) # present the sequence seq_len timesteps at a time
        lstm_out, self.hidden = self.lstm(input, self.hidden)
        scores = self.linear(lstm_out.view(self.seq_len,-1))
        return scores

def train_seq(X_train, y_train, gpu_enabled=False, verbose=True):
    for i in range(X_train.shape[0]):
        for j in range(X_train.shape[1]):
            # model.init_hidden(hidden_dim=20)
            label = y_train[i][j]
            if gpu_enabled:
                label = autograd.Variable(torch.LongTensor([int(label % 769)] * 1000).cuda())
            else:
                label = autograd.Variable(torch.LongTensor([int(label % 769)] * 1000).cuda())
            sample = X_train[i][j]
            if np.any(np.isnan(sample)):
                print("skipping sample with NaN")
                continue
            assert not np.any(np.isnan(sample))
            model.zero_grad()
            scores = model(sample.cuda())
            loss = loss_function(scores.cuda(), label.cuda())
            loss.backward(retain_graph = True)
            optimizer.step()
             # clip the gradient to prevent exploding gradients (probably not needed)
            nn.utils.clip_grad_norm(model.parameters(), 0.99)
            if verbose:
                print(loss.data[0])
        break

def train_one_by_one(model, loss_function, optimizer, X_train, y_train, gpu_enabled=False, verbose=True):
    i, j = 0, 0
    iter_count = 0
    for i in range(X_train.shape[0]):
        if verbose: print("training on dataset: {}".format(i))
        for j in range(X_train.shape[1]):
            if verbose: print("training on trial {}".format(j))
            sample = X_train[i][j].T
            print(sample.shape)
            label = y_train[i][j]
            # hack: deal with NaN
            if np.any(np.isnan(sample)):
                print("skipping sample with NaNs")
                continue
            assert not np.any(np.isnan(sample))
            if gpu_enabled:
                label = autograd.Variable(torch.LongTensor([int(label % 769)]).cuda())
            else:
                label = autograd.Variable(torch.LongTensor([int(label % 769)]))
#            model.hidden = model.init_hidden(20)
            accumulated_loss = 0
            # present the seq across 1k timesteps, building up the state one at a time
            for k in range(sample.shape[0]):
                model.zero_grad()
                input = sample[0]
                assert input.shape[0] == 22
                scores = model(input)
                loss = loss_function(scores.cuda() if gpu_enabled else scores, label.cuda() if gpu_enabled else label)
                accumulated_loss +=loss.data[0]
                out = loss.backward(retain_graph = True)
                optimizer.step()
                # clip the gradient to prevent exploding gradients (probably not needed)
                nn.utils.clip_grad_norm(model.parameters(), 0.99)
                if k % 50 == 0 and verbose: print("loss at timestep {}: {}".format(k, loss.data[0]))
            lastpred = np.argmax(scores.cpu().data.numpy().reshape(4))
            # save the model after every 1k timesteps
            print('saving model weights to path ./model.dat')
            torch.save(model.state_dict(), 'model.dat')
            if verbose: print("Predicted label at last timestep: {}, actual label: {}".format(lastpred, y_train[i][j] % 769))
            if verbose: print("average loss after from previous 1000 timesteps: {}".format(accumulated_loss/1000))


# prediction code
def predict():
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the LSTM on EEG data')
    parser.add_argument('--bidirectional', action='store_true', default=False, help='Make LSTM bidirectional.')
    parser.add_argument('--use-gpu', dest='use_gpu', action='store_true', default=False, help='CUDA enabled or not.')
    parser.add_argument('--no-verbose', dest='no_verbose', action='store_true', default=False, help='Turn off verbose logging.')
    parser.add_argument('--lr', dest='lr', type=float, default = 0.0001, help='Optimizer learning rate hyperparameter.')
    parser.add_argument('--load', dest='load', type=str, default=None, help='Path to pretrained model weights.')
    args = parser.parse_args()
    print('----MODEL ARGUMENTS------')
    print("Using GPU: {}".format(args.use_gpu))
    print("LSTM bidirectional: {}".format(args.bidirectional))
    print("Learning rate: {}".format(args.lr))
    print("Verbosity: {}".format(not args.no_verbose))
    print('Path to pretrained: {}'.format(args.load))
    print('initializing model')
    model = EEGLSTM(seq_len=1, input_dim=22, hidden_dim=20, output_dim=4, batch_size = 1, bidirectional=args.bidirectional, gpu_enabled=args.use_gpu) # seq len, input dim, hidden dim, output dim, biirectional
    # .cuda() if use gpu
    if args.use_gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    # model.load_state_dict(torch.load(PATH))
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    data_loader = EEGDataLoader('project_datasets/')
    print('loading data')
    X_train, y_train, X_test, y_test = data_loader.load_all_data()
    train_one_by_one(model, loss_function, optimizer, X_train, y_train, gpu_enabled=args.use_gpu, verbose=not args.no_verbose)
