import h5py
import numpy as np
import torch
import random
from tokenizer import tokenize
from torch.utils.data import Dataset, DataLoader


TRAIN_DATA_PATH = '/data/hamaraa/mate_in_k_train_1m.h5'
VAL_DATA_PATH = '/data/hamaraa/mate_in_k_val.h5'
NUM_POSITIVES = 5

class MateInKDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path

        with h5py.File(file_path, 'r') as f:
            self.fens = np.array(f['fens'])
            self.ks = np.array(f['k'])

        self.k_to_indices = {}
        for idx, k in enumerate(self.ks):
            if k not in self.k_to_indices:
                self.k_to_indices[k] = []
            self.k_to_indices[k].append(idx)

    def __len__(self):
        return len(self.ks)

    def __getitem__(self, idx):
        anchor_token_tensor = torch.tensor(self.fens[idx], dtype=torch.int32)
        anchor_k = self.ks[idx]
        
        pos_indices = self.k_to_indices[anchor_k]
        p = min(len(pos_indices), NUM_POSITIVES)
        pos_i = np.random.choice(pos_indices, p, replace=False)

        positive_token_tensors = torch.stack([
            torch.tensor(self.fens[i], dtype=torch.int32)
            for i in pos_i
        ])

        return (anchor_token_tensor, positive_token_tensors), anchor_k

    def close(self):
        self.f.close()


def get_dataloader(batch_size=512, shuffle=True, num_workers=32, split='train'):
    if split == 'train':
        dataset = MateInKDataset(TRAIN_DATA_PATH)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
        return dataloader
    elif split == 'val':
        dataset = MateInKDataset(VAL_DATA_PATH, num_negatives=32, num_positives=8)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
        return dataloader
