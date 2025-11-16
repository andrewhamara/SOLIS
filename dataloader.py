import h5py
import numpy as np
from tokenizer import tokenize
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import random
import torch

TRAIN_DATA_PATH = '/data/hamaraa/tokenized_5m.h5'

class SolisDataset(Dataset):
    def __init__(self, h5_path, k_pos, indices=None, p_threshold=.05):
        self.h5_path = h5_path
        self.k_pos = k_pos
        self.p_threshold = p_threshold

        # open file and load arrays into memory (optional: mmap if large)
        with h5py.File(h5_path, 'r') as f:
            all_tokens = np.array(f['fens'])
            all_ps = np.array(f['ps'])

        if indices is None:
            indices = np.arange(len(all_ps))
        self.indices = np.array(indices)

        self.tokens = all_tokens[self.indices]
        self.ps = all_ps[self.indices]

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

def split_data(h5_path, train_ratio=0.8, seed=0):
    with h5py.File(h5_path, 'r') as f:
        n = len(f['ps'])
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    cut = int(train_ratio * n)
    return indices[:cut], indices[cut:]

def get_dataloader(batch_size=512, shuffle=True, num_workers=32, split='train', k_pos=5):
    train_idx, val_idx = split_data(TRAIN_DATA_PATH)
    if split == 'train':
        dataset = SolisDataset(TRAIN_DATA_PATH, k_pos, indices=train_idx)
    else:
        dataset = SolisDataset(TRAIN_DATA_PATH, k_pos, indices=val_idx)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'), num_workers=num_workers, pin_memory=True)
    return dataloader