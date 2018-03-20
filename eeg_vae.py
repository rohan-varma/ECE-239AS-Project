#CODE Adapted from https://github.com/pytorch/examples/blob/master/vae/main.py
from __future__ import print_function
import argparse
import torch
import matplotlib.pyplot as plt
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from load_data import EEGDataLoader
import numpy as np
import os

parser = argparse.ArgumentParser(description='EEG VAE')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self, input_dim):
        super(VAE, self).__init__()
        self.indim = input_dim
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, input_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.indim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE(22*1000)
data_loader = EEGDataLoader('project_datasets/')
print('loading data')
X_train, y_train, X_test, y_test = data_loader.load_all_data()
if args.cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    mse = torch.nn.MSELoss(size_average = False)
    BCE = mse(recon_x, x.view(-1, 22*1000))
#    BCE = F.nll_loss(recon_x, x.view(-1, 22*1000), size_average=False)
    if BCE.data[0] < 0:
        print('ALERT: BCE negative')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def plot(x, recon_x,i):
    x = x.data.numpy().reshape((22, 1000))
    recon_x = recon_x.data.numpy().reshape((22, 1000))
    single_x, single_recon_x = x[0], recon_x[0]
    fig = plt.figure()
    plt.title('Electrode 0 values across 313 timesteps')
    plt.plot(range(0, single_x.shape[0]), single_x)
    plt.savefig('figures/x_{}.png'.format(i))
    plt.close(fig)

    fig = plt.figure()
    plt.title('Electrode 0 values across 313 timesteps')
    plt.plot(range(0, single_recon_x.shape[0]), single_recon_x)
    plt.savefig('figures/recon_x_{}.png'.format(i))
    plt.close(fig)


def train(epoch):
    model.train()
    train_loss = 0
    for i in range(X_train.shape[0]):
        cur_X = X_train[i]
        for j in range(cur_X.shape[0]):
            batch_idx, data = j, cur_X[j]
            data = (data - np.mean(data, axis = 0))/(np.std(data, axis = 0) + np.finfo(float).eps)
            data = torch.FloatTensor(data)
            data = Variable(data)
            if args.cuda:
                data = data.cuda()
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            # plotting electrode 0 across 1k
            #print('about to plot and save')
            #plot(data, recon_batch, j)
            loss = loss_function(recon_batch, data, mu, logvar)
            print(recon_batch.shape)
            loss.backward()
            train_loss += loss.data[0]
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), i,
                    100. * batch_idx / len(train_loader),
                    loss.data[0] / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    print('----TESTING----')
    model.eval()
    test_loss = 0
    for k in range(X_train.shape[0]):
        cur_X = X_train[k]
        for i in range(cur_X.shape[0]):
            data = cur_X[i]
            data = (data - np.mean(data, axis = 0))/(np.std(data, axis = 0) + np.finfo(float).eps)
            data = torch.FloatTensor(data)
            if args.cuda:
                data = data.cuda()
            data = Variable(data, volatile=True)
            recon_batch, mu, logvar = model(data)
            print(recon_batch.shape)
            print('Plotting data....view it in figures/ directory')
            plot(data, recon_batch, i) # plot the originals and reconstructions
            test_loss += loss_function(recon_batch, data, mu, logvar).data[0]
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(1, 1, 22, 1000)[:n]])
                save_image(comparison.data.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    sample = Variable(torch.randn(64, 20))
    if args.cuda:
        sample = sample.cuda()
    sample = model.decode(sample).cpu()
    save_image(sample.data.view(64, 1, 22, 1000),
               'results/sample_' + str(epoch) + '.png')