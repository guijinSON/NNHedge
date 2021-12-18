import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class MLPNet(nn.Module):
    def __init__(self, span_length):
        super(MLPNet, self).__init__()
        self.lin1 = nn.Linear(span_length, 32)
        self.tanh1 = nn.Tanh()

        self.lin2 = nn.Linear(32, 4)
        self.tanh2 = nn.Tanh()

        self.lin3 = nn.Linear(4,1)
        self.tanh = nn.Tanh()
        
        self.span_length = span_length

    def forward(self, X):
        X = torch.add(X,torch.range(start=0,end=self.span_length-1,step=1))
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
    
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    
    def forward(self, X):
        return X[:, :, : -self.chomp_size].contiguous()
    
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
 
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.3, emb_dropout=0.1):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear= nn.Linear(num_channels[-1], output_size)
        self.tanh = nn.Tanh()

    def forward(self, X):
        X = self.tcn(X)[:, :, -1]
        output = self.tanh(self.linear(X))
        return output

class AttentionNet(nn.Module):
    def __init__(self, span_length):
        super(AttentionNet, self).__init__()
        self.query = nn.Linear(span_length,10)
        self.value = nn.Linear(span_length,10)

        self.FFN = nn.Linear(10,1)
        self.tanh = nn.Tanh()
        
        self.span_length = span_length
    
    def forward(self,X):
        X = torch.add(X,torch.range(start=0,end=self.span_length-1,step=1))

        q = self.query(X)
        v = self.query(X)

        weight = F.softmax(q, dim=-1)
        X = torch.mul(weight,v)
        X = self.tanh(self.FFN(X))

        return X
