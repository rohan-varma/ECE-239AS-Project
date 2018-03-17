
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
        # input, hidden size
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
            return (autograd.Variable(torch.zeros(1 * bidirectional_mult, self.batch_size, hidden_dim)).cuda(),
                autograd.Variable(torch.zeros(1 * bidirectional_mult, self.batch_size, hidden_dim)).cuda())
        else:
            return (autograd.Variable(torch.zeros(1 * bidirectional_mult, self.batch_size, hidden_dim)),
                autograd.Variable(torch.zeros(1 * bidirectional_mult, self.batch_size, hidden_dim)))
    
    def forward(self, input):
        """forwards input through the model"""
        # convert the input into something Pytorch can understand
        if self.gpu_enabled:
            input = autograd.Variable(torch.FloatTensor(input)).contiguous().cuda()
        else:
            input = autograd.Variable(torch.FloatTensor(input)).contiguous()
        # LSTM expects 3-D input: dim of input is expected to be seq_len, batch size, input size
        input = input.view(self.seq_len, self.batch_size, -1) # present the sequence seq_len timesteps at a time
        lstm_out, self.hidden = self.lstm(input, self.hidden)
        scores = self.linear(lstm_out.view(self.seq_len, self.batch_size, -1)).view(self.batch_size, self.output_dim)
        return scores

def train_seq(model, loss_function, optimizer, X_train, y_train, gpu_enabled=False, verbose=True):
    for i in range(X_train.shape[0]):
        for j in range(X_train.shape[1]):
            # model.init_hidden(hidden_dim=20)
            label = y_train[i][j]
            if gpu_enabled:
                label = autograd.Variable(torch.LongTensor([int(label % 769)] * 1000).cuda())
            else:
                label = autograd.Variable(torch.LongTensor([int(label % 769)] * 1000))
            sample = X_train[i][j]
            if np.any(np.isnan(sample)):
                print("skipping sample with NaN")
                continue
            assert not np.any(np.isnan(sample))
            optimizer.zero_grad()
            scores = model(sample.cuda() if gpu_enabled else sample)
            loss = loss_function(scores.cuda() if gpu_enabled else scores, label.cuda() if gpu_enabled else label)
            loss.backward(retain_graph = True)
            optimizer.step()
             # clip the gradient to prevent exploding gradients (probably not needed)
            nn.utils.clip_grad_norm(model.parameters(), 0.99)
            if verbose:
                print(loss.data[0])
        break

def train_batch(model, loss_function, optimizer, X_train, y_train, gpu_enabled=False, verbose=True, batch_size = 50):
    print("shapes: {}".format(X_train.shape, y_train.shape))
    print(X_train[0].shape)
    exit()
    j = 0
    max_j = X_train.shape[1]
    assert X_train.shape[1] == y_train.shape[1], "{} {}".format(X_train.shape[1], y_train.shape[1])
    for i in range(X_train.shape[0]):
        # add verbose logging
        if j + batch_size >= max_j:
            break
        batch = X_train[i][j:j+batch_size]
        labels = y_train[i][j:j+batch_size]
        if np.any(np.isnan(batch)):
            print('skipping sample with nans')
            continue
        j+=batch_size
        assert not np.any(np.isnan(batch))
        labels = autograd.Variable(torch.LongTensor([int(label % 769) for label in list(labels)]))
        accumulated_loss = 0
        hidden = model.init_hidden(20)
        for k in range(batch.shape[2]):
            optimizer.zero_grad()
            input = batch[:,:,k]
            scores = model(input)
            loss = loss_function(scores.cuda() if gpu_enabled else scores, labels.cuda() if gpu_enabled else labels)
            accumulated_loss+=loss.data[0]
            out = loss.backward(retain_graph=True)
            optimizer.step()
            nn.utils.clip_grad_norm(model.parameters(), 0.99)
            if k % 50 == 0 and verbose: print("loss at timestep {}: {}".format(k, loss.data[0]))
        print('average loss frm previous 1000 timesteps: {}'.format(accumulated_loss/batch.shape[2]))

        
        scores = model(batch[:,:,i])
        loss = loss_function(scores.cuda() if gpu_enabled else scores, labels.cuda() if gpu_enabled else labels)
        print(loss.data[0])
        print(scores.shape)
        exit()

def detach(states):
    return [state.detach() for state in states] 

def train_one_by_one(model, loss_function, optimizer, X_train, y_train, gpu_enabled=False, verbose=True):
    i, j = 0, 0
    iter_count = 0
    accs = []
    assert(X_train.shape[0] == 9)
    for i in range(X_train.shape[0]):
        X_s, y_s = X_train[i], y_train[i]
        for j in range(X_s.shape[0]):
            trial, label = X_s[j], y_s[j]
            if gpu_enabled:
                label = autograd.Variable(torch.LongTensor([int(label) % 769]).cuda())
            else:
                label = autograd.Variable(torch.LongTensor([int(label) % 769]))
            # re init the hidden state before we go across tiemsteps
            model.hidden = model.init_hidden(20)
            acc_loss = 0
            # present the sequence one at a time
            for k in range(trial.shape[1]):
                optimizer.zero_grad()
                #bptt only 100 timesteps
                if k % 100 == 0:
                    model.hidden = detach(model.hidden)
                input = trial[:,k]
                scores = model(input)
                loss = loss_function(scores.cuda() if gpu_enabled else scores, label.cuda() if gpu_enabled else label)
                acc_loss+=loss.data[0]
                loss.backward(retain_graph=True)
                optimizer.step()
                if verbose and k % 100 == 0:
                    # report the loss and make a prediction
                    pred = np.argmax(scores.cpu().data.numpy().reshape(4))
                    print("Timestep: {}, loss: {}, prediction: {}, actual: {}".format(k, loss.data[0], pred + 769, label.data[0] + 769))
            # # now that we're done presenting 1k timesteps, get the last prediction
            lastpred = np.argmax(scores.cpu().data.numpy().reshape(4))
            print("After 1k timesteps: pred {} actual {} loss {}".format(lastpred + 769, label.data[0] + 769, loss.data[0])) 
        accs.append(accuracy_single_subject(X_s, y_s))
    print("subject accuracies: {}, mean is {}".format(accs, np.mean(np.array(accs))))


def accuracy_single_subject(X_s, y_s):
    # forward through the model and get the acc across a single subject
    # go across the trials
    predictions = []
    for j in range(X_s.shape[0]):
        trial, label = X_s[j], y_s[j]
        label = autograd.Variable(torch.LongTensor([int(label) % 769]))
        # re init the hidden state before we go across tiemsteps
        model.hidden = model.init_hidden(20)
        acc_loss = 0
        # present the sequence one at a time
        for k in range(trial.shape[1]):
            #optimizer.zero_grad()
            model.hidden = detach(model.hidden) # not sure if I need this line or not
            input = trial[:,k]
            scores = model(input)
        lastpred = np.argmax(scores.cpu().data.numpy().reshape(4))
        predictions.append(lastpred)
        print("Evaluating accuracy: pred {} actual {}".format(lastpred + 769, label.data[0] + 769)) 
    errs = [1 if (p+769) != l else 0 for p,l in zip(predictions, y_s)]
    error_rate = sum(errs)/len(errs)
    print("Accuracy for this subject: {}".format(1 - error_rate))
    return 1 - error_rate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the LSTM on EEG data')
    parser.add_argument('--bidirectional', action='store_true', default=False, help='Make LSTM bidirectional.')
    parser.add_argument('--use-gpu', dest='use_gpu', action='store_true', default=False, help='CUDA enabled or not.')
    parser.add_argument('--no-verbose', dest='no_verbose', action='store_true', default=False, help='Turn off verbose logging.')
    parser.add_argument('--lr', dest='lr', type=float, default = 0.0001, help='Optimizer learning rate hyperparameter.')
    parser.add_argument('--load', dest='load', type=str, default=None, help='Path to pretrained model weights.')
    parser.add_argument('--seq-len', dest='seq_len', type=int, default=1, help ='Sequence length from 1 to 1k')
    parser.add_argument('--batch', type=int, default=1, help='Batch size to use (like 250)')
    parser.add_argument('--single-subject', dest='single_subject', action='store_true', default=False,
        help='Train only on a single subject.')

    args = parser.parse_args()
    print('----MODEL ARGUMENTS------')
    print("Using GPU: {}".format(args.use_gpu))
    print("LSTM bidirectional: {}".format(args.bidirectional))
    print("Learning rate: {}".format(args.lr))
    print("Verbosity: {}".format(not args.no_verbose))
    print('Path to pretrained: {}'.format(args.load))
    print('Sequence length: {}'.format(args.seq_len))
    print('Batch size: {}'.format(args.batch))
    print('initializing model')
    model = EEGLSTM(seq_len=args.seq_len, input_dim=22, hidden_dim=20, output_dim=4, batch_size = args.batch, bidirectional=args.bidirectional, gpu_enabled=args.use_gpu) # seq len, input dim, hidden dim, output dim, biirectional
    # .cuda() if use gpu
    if args.use_gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        model.cuda()
       # model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    # model.load_state_dict(torch.load(PATH))
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    data_loader = EEGDataLoader('project_datasets/')
    print('loading data')
    X_train, y_train, X_test, y_test = data_loader.load_all_data()
    print('Beginning training: sequence length of {}'.format(args.seq_len))
    if args.seq_len == 1:
        # train one by one
        if args.batch != 1:
            print("USING BATCH SIZE {}".format(args.batch))
            train_batch(model, loss_function, optimizer, X_train, y_train, gpu_enabled=args.use_gpu, verbose=not args.no_verbose, batch_size = args.batch)
        else:
            train_one_by_one(model, loss_function, optimizer, X_train, y_train, gpu_enabled=args.use_gpu, verbose=not args.no_verbose)

    else:
        train_seq(model, loss_function, optimizer, X_train, y_train, gpu_enabled=args.use_gpu, verbose=not args.no_verbose)
