import h5py
import numpy as np
import torch
import random
from tokenizer import tokenize
from torch.utils.data import Dataset, DataLoader


TRAIN_DATA_PATH = '/data/hamaraa/mate_in_k_train_500k.h5'
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
 
        self.k_neg_indices = {
            k: np.where(self.ks != k)[0] for k in np.unique(self.ks)
        }

    def __len__(self):
        return len(self.ks)

    def __getitem__(self, idx):
        anchor_token_tensor = torch.tensor(self.fens[idx], dtype=torch.int32)
        anchor_k = self.ks[idx]

        # same k as anchor
        pos_indices = self.k_to_indices[anchor_k]
        pos_i = random.choice(pos_indices) if len(pos_indices) > 1 else idx
        positive_token_tensor = torch.tensor(self.fens[pos_i], dtype=torch.int32)

        #pos_fens = torch.stack([
        #    torch.tensor(tokenize(self.fens[i].decode('utf-8')), dtype=torch.int32)
        #    for i in pos_i
        #])

        # all other k
        neg_indices = self.k_neg_indices[anchor_k]
        num_neg_samples = min(self.num_negatives, len(neg_indices))
        neg_i = np.random.choice(neg_indices, num_neg_samples, replace=False)

        negative_token_tensors = torch.stack([
            torch.tensor(self.fens[i], dtype=torch.int32)
            for i in neg_i
        ])

        return anchor_token_tensor, positive_token_tensor, negative_token_tensors
        #return {
        #    'anchor': torch.tensor(anchor_fen, dtype=torch.int32),
        #    'positives': pos_fens,
        #    'negatives': neg_fens,
        #    'anchor_k': anchor_k
        #}


    def close(self):
        self.f.close()


def get_dataloader(batch_size=512, shuffle=True, num_workers=32, split='train'):
    if split == 'train':
        dataset = MateInKDataset(TRAIN_DATA_PATH, num_negatives=8, num_positives=8)
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
