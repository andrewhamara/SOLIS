import h5py
import numpy as np
from tokenizer import tokenize
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import random
import torch

TRAIN_DATA_PATH = '/data/hamaraa/tokenized_5m.h5'

class SolisDataset(Dataset):
    def __init__(self, h5_path, k_pos, p_threshold=.05):
        self.h5_path = h5_path
        self.k_pos = k_pos
        self.p_threshold = p_threshold

        # open file and load arrays into memory (optional: mmap if large)
        with h5py.File(h5_path, 'r') as f:
            self.tokens = np.array(f['fens'])
            self.ps = np.array(f['ps'])
 
        self.sorted_indices = np.argsort(self.ps)
        self.sorted_ps = self.ps[self.sorted_indices]


    def __len__(self):
        return len(self.ps)

    def __getitem__(self, idx):
        anchor_tensor = torch.tensor(self.tokens[idx], dtype=torch.int32)
        anchor_p = self.ps[idx]

        lower = np.searchsorted(self.sorted_ps, anchor_p - self.p_threshold, side='left')
        upper = np.searchsorted(self.sorted_ps, anchor_p + self.p_threshold, side='right')

        candidate_i = self.sorted_indices[lower:upper]
        candidate_i = candidate_i[candidate_i != idx]

        sampled_i = np.random.choice(candidate_i, self.k_pos, replace=False)

        positives = [torch.tensor(self.tokens[i], dtype=torch.int32) for i in sampled_i]
        pos_tensor_stack = torch.stack(positives)

        return {
            "anchor": anchor_tensor,
            "positives": pos_tensor_stack,
            "label": torch.tensor(anchor_p, dtype=torch.float32)
        }

def get_dataloader(batch_size=512, shuffle=True, num_workers=32, split='train', k_pos=5):
    if split == 'train':
        dataset = SolisDataset(TRAIN_DATA_PATH, k_pos)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
        return dataloader
    #elif split == 'val':
    #    dataset = SolisDataset(VAL_DATA_PATH, num_negatives=32, num_positives=8)
    #    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    #    return dataloader
