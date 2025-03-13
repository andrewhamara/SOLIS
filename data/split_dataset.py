import h5py
import numpy as np

DATA_PATH = '/data/hamaraa/mate_in_k_dataset.h5'
TRAIN_PATH = '/data/hamaraa/mate_in_k_train.h5'
VAL_PATH = '/data/hamaraa/mate_in_k_val.h5'
TEST_PATH = '/data/hamaraa/mate_in_k_test.h5'

# load
with h5py.File(DATA_PATH, 'r') as f:
    fens = np.array(f['fens'])
    ks = np.array(f['k'])

# shuffle
num_samples = len(fens)
indices = np.arange(num_samples)
np.random.shuffle(indices)

# split
train_split = int(0.6 * num_samples)
val_split = int(0.8 * num_samples)

train_indices = indices[:train_split]
val_indices = indices[train_split:val_split]
test_indices = indices[val_split:]

def save_h5(filename, fens_subset, ks_subset):
    with h5py.File(filename, 'w') as f:
        f.create_dataset('fens', data=fens_subset, compression='gzip')
        f.create_dataset('k', data=ks_subset, compression='gzip')
    print(f"Saved {len(fens_subset)} samples to {filename}")

save_h5(TRAIN_PATH, fens[train_indices], ks[train_indices])
save_h5(VAL_PATH, fens[val_indices], ks[val_indices])
save_h5(TEST_PATH, fens[test_indices], ks[test_indices])
