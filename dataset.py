import h5py
import torch
import numpy as np
from torch.utils import data


class Pcam(data.Dataset):

    def __init__(self, x_archive, y_archive, transform=None):
        self.x_archive = h5py.File(x_archive, 'r', swmr=True)
        self.y_archive = h5py.File(y_archive, 'r', swmr=True)
        self.data = self.x_archive['x']
        self.labels = self.y_archive['y']
        self.transform = transform

    def __getitem__(self, item):
        x = self.data[item]
        if self.transform is not None:
            x = self.transform(x)
        return x, np.squeeze(self.labels[item])

    def __len__(self):
        return len(self.labels)

    def close(self):
        self.x_archive.close()
        self.y_archive.close()
