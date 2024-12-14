import numpy as np
import torch
from torch.utils.data import Dataset

class FWC2Dataset(Dataset):
    def __init__(self, data, target, columns=None, labels=None):
        self.data = np.array(data)
        self.target = np.array(target)
        self.columns = columns
        self.labels = labels

    @property
    def features_low(self):
        return self.data.min(axis=0)

    @property
    def features_high(self):
        return self.data.max(axis=0)

    @property
    def shape(self):
        return self.data.shape

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float32), torch.tensor(self.target[index], dtype=torch.long)

    def __len__(self):
        return len(self.data)
