class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1,25, kernel_size=10),
            nn.Conv1d(25,25, kernel_size = 1),
            nn.BatchNorm1d(25),
            nn.ELU(),
            nn.MaxPool1d(15))
        self.fc1 = nn.Sequential(
            nn.Linear(1650,4))
        
    def forward(self, x):
        out = self.conv1(x)
        C, E, T = out.size()
        out = out.view(C, -1)
       # print(out.size())
        out = self.fc1(out)
        return out
        
        