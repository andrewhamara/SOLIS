import h5py
import numpy as np
import torch

DATA_PATH = '/data/hamaraa/mate_in_k_dataset.h5'
CLIP_RANGE = 30

def clip_data():
    with h5py.File(DATA_PATH, 'r+') as f:
        ks = np.array(f['k'])
        fens = np.array(f['fens'])

        valid_i = (ks >= -CLIP_RANGE) & (ks <= CLIP_RANGE) | (ks == 444) | (ks == -444)

        # mask out checkmates 
        ks_filtered = ks[valid_i]
        fens_filtered = fens[valid_i]

        unique_fens, unique_indices = np.unique(fens_filtered, return_index=True)
        ks_unique = ks_filtered[unique_indices]

        del f['k'], f['fens']
        f.create_dataset('k', data=ks_filtered)
        f.create_dataset('fens', data=unique_fens)

        print(f'removed {len(ks) - len(ks_filtered)} samples from clipping')
        print(f'removed {len(fens_filtered) - len(unique_fens)} duplicates after clipping')

if __name__ == '__main__':
    clip_data()
