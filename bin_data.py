import h5py
import numpy as np
from tqdm import tqdm

input_path = "/data/hamaraa/tokenized_1m.h5"
output_path = "/data/hamaraa/tokenized_and_binned_1m.h5"

# bin config
bin_edges = np.arange(0.0, 1.0001, 0.05)
bin_labels = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(len(bin_edges) - 1)]


# load tokenized data
with h5py.File(input_path, 'r') as f:
    tokens = f['fens'][:]
    ps = f['ps'][:]

# bin win percentages
bin_indices = np.digitize(ps, bin_edges, right=False) - 1
bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)

counts = np.bincount(bin_indices, minlength=len(bin_labels))
print("Bin Distribution:")
for label, count in zip(bin_labels, counts):
    print(f"{label}: {count}")
print()

# save with bins
with h5py.File(output_path, 'w') as f:
    f.create_dataset('tokens', data=tokens, compression='gzip')
    f.create_dataset('ps', data=ps, compression='gzip')
    f.create_dataset('bins', data=bin_indices, compression='gzip')
