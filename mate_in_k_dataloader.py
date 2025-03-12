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
        self.f = h5py.File(file_path, 'r')

        self.fens = np.array(self.f['fens'])
        self.ks = np.array(self.f['k'])

    def __len__(self):
        return len(self.ks)
    
    def __getitem__(self, idx):
        anchor_fen = self.fens[idx].decode('utf-8')
        anchor_fen = tokenize(anchor_fen)
        anchor_k = self.ks[idx]

        pos_i = np.where(self.ks == anchor_k)[0]
        pos_i = random.choice(pos_i) if len(pos_i) > 1 else idx
        pos_fen = self.fens[idx].decode('utf-8')
        pos_fen = tokenize(pos_fen)

        neg_i = np.where(self.ks != anchor_k)[0]
        num_neg_samples = min(self.num_negatives, len(neg_i))
        neg_i = np.random.choice(neg_i, num_neg_samples, replace=False)

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
