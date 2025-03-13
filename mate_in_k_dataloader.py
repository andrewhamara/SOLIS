import h5py
import numpy as np
import torch
import random
from tokenizer import tokenize
from torch.utils.data import Dataset, DataLoader


TRAIN_DATA_PATH = '/data/hamaraa/mate_in_k_train.h5'
VAL_DATA_PATH = '/data/hamaraa/mate_in_k_val.h5'

class MateInKDataset(Dataset):
    def __init__(self, file_path, num_negatives, num_positives):
        self.num_negatives = num_negatives
        self.num_positives = num_positives
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
        anchor_fen = self.fens[idx].decode('utf-8')
        anchor_fen = tokenize(anchor_fen)
        anchor_k = self.ks[idx]

        # same k as anchor
        #print('getting positives')
        pos_indices = self.k_to_indices[anchor_k]
        num_pos_samples = min(self.num_positives, len(pos_indices))
        pos_i = np.random.choice(pos_indices, num_pos_samples, replace=False)

        pos_fens = torch.stack([
            torch.tensor(tokenize(self.fens[i].decode('utf-8')), dtype=torch.int32)
            for i in pos_i
        ])

        # all other k
        #print('getting negatives')
        neg_mask = np.abs(self.ks - anchor_k) >= 1
        neg_indices = np.where(neg_mask)[0]
        num_neg_samples = min(self.num_negatives, len(neg_indices))
        neg_i = np.random.choice(neg_indices, num_neg_samples, replace=False)

        neg_fens = torch.stack([
            torch.tensor(tokenize(self.fens[i].decode('utf-8')), dtype=torch.int32)
            for i in neg_i
        ])

        return anchor_fen, pos_fens, neg_fens


    def close(self):
        self.f.close()


def get_dataloader(batch_size=512, shuffle=True, num_workers=32, split='train'):
    if split == 'train':
        dataset = MateInKDataset(TRAIN_DATA_PATH, num_negatives=32, num_positives=8)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
        return dataloader
    elif split == 'val':
        dataset = MateInKDataset(VAL_DATA_PATH, num_negatives=32, num_positives=8)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
        return dataloader


if __name__ == '__main__':
    dataloader = get_dataloader()

    for batch in dataloader:
        fens, ks = batch
