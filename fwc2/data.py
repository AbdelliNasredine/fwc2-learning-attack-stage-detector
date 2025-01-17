import numpy as np
import torch
from torch.utils.data import Dataset

class NetFlowDataset(Dataset):
    def __init__(
            self,
            data,
            labels=None,
            columns=None,
    ):
        self.data = np.array(data)
        self.labels = np.array(labels)
        self.columns = columns

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
        return torch.tensor(self.data[index], dtype=torch.float32), torch.tensor(self.labels[index], dtype=torch.long)

    def __len__(self):
        return len(self.data)
