import torch 
import torch.nn as nn

span_length = 3

class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.lin1 = nn.Linear(span_length, 32)
        self.tanh1 = nn.Tanh()

        self.lin2 = nn.Linear(32, 4)
        self.tanh2 = nn.Tanh()

        self.lin3 = nn.Linear(4,1)
        self.tanh = nn.Tanh()

    def forward(self, X):
        X = torch.add(X,torch.tensor([0,1,2]))
        X = self.tanh1(self.lin1(X))
        X = self.tanh2(self.lin2(X))
        out = self.tanh(self.lin3(X))
        return out    


class RNNNet(nn.Module):
    def __init__(self):
        super(RNNNet, self).__init__()
        self.rnn = nn.RNN(1,5,3,batch_first=True) 
        self.linear = nn.Linear(5,1)
        self.tanh = nn.Tanh()

    def forward(self, X):
        X = self.rnn(X)[0]
        X = self.tanh(self.linear(X))
        out = X[:,-1]        
        return out    
