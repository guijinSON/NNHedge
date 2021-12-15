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

def get_dataloader(data, batch_size = 256, shuffle=True, drop_last=True):
    return DataLoader(data, batch_size = batch_size, shuffle = shuffle, drop_last = drop_last)
