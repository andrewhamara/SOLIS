import h5py
import numpy as np
import torch
import random
from tokenizer import tokenize
from torch.utils.data import Dataset, DataLoader


DATA_PATH = '/data/hamaraa/mate_in_k_test.h5'

class MateInKDataset(Dataset):
    def __init__(self, file_path, num_negatives):
        self.num_negatives = num_negatives
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

        pos_indices = self.k_to_indices[anchor_k]
        pos_i = random.choice(pos_indices) if len(pos_indices) > 1 else idx
        pos_fen = self.fens[pos_i].decode('utf-8')
        pos_fen = tokenize(pos_fen)

        neg_indices = np.where(self.ks != anchor_k)[0]
        num_neg_samples = min(self.num_negatives, len(neg_indices))
        neg_i = np.random.choice(neg_indices, num_neg_samples, replace=False)

        neg_fens = torch.stack([
            torch.tensor(tokenize(self.fens[i].decode('utf-8')), dtype=torch.int32)
            for i in neg_i
        ])

        return anchor_fen, pos_fen, neg_fens


    def close(self):
        self.f.close()


def get_dataloader(batch_size=512, shuffle=True, num_workers=4):
    dataset = MateInKDataset(DATA_PATH, 20)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return dataloader


if __name__ == '__main__':
    dataloader = get_dataloader()

    for batch in dataloader:
        fens, ks = batch
