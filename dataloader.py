import torch 
from torch.utils.data import Dataset, DataLoader

class SpanDataset(Dataset):
    def __init__(self,span_X,span_C):
        self.span_X = span_X
        self.span_C = span_C

    def __len__(self):
        return len(self.span_X)

    def __getitem__(self, idx):
        X = self.span_X[idx]
        Sn = X[-1]
        Sn_1 = X[-2]
        C = self.span_C[idx]
        return X, C, Sn, Sn_1
