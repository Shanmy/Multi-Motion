from torch.utils.data import Dataset
import numpy as np


class DummyDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        return np.ones((158)), {'texts': 'aaaaa'}  # mask: batch_size x seq_len
