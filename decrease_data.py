import h5py
import numpy as np
import torch

FULL_DATA_PATH = '/data/hamaraa/mate_in_k_train_tokenized.h5'
SUBSET_DATA_PATH = '/data/hamaraa/mate_in_k_train_10k.h5'
TARGET_SIZE = 10_000

print("Loading full dataset...")
with h5py.File(FULL_DATA_PATH, 'r') as f:
    fens = np.array(f['fens'])
    ks = np.array(f['k'])

# Step 1: Compute distribution
unique_ks, counts = np.unique(ks, return_counts=True)
total_samples = len(ks)

# Step 2: Compute target sample size per k
sampling_ratios = counts / total_samples  # Fraction of each k
subset_sizes = (sampling_ratios * TARGET_SIZE).astype(int)  # Scale to 500k

# Ensure the total matches exactly 500,000
diff = TARGET_SIZE - subset_sizes.sum()
if diff > 0:
    subset_sizes[np.argsort(-counts)[:diff]] += 1  # Add to most common ks
elif diff < 0:
    subset_sizes[np.argsort(counts)[:abs(diff)]] -= 1  # Remove from rare ks

assert subset_sizes.sum() == TARGET_SIZE, "Subset sizes don't add up!"

# Step 3: Sample proportionally
selected_indices = []
for k, num_samples in zip(unique_ks, subset_sizes):
    k_indices = np.where(ks == k)[0]
    selected = np.random.choice(k_indices, num_samples, replace=False)
    selected_indices.extend(selected)

selected_indices = np.array(selected_indices)

# Step 4: Save to new HDF5 file
print(f"Saving {TARGET_SIZE} samples to {SUBSET_DATA_PATH}...")
with h5py.File(SUBSET_DATA_PATH, 'w') as f:
    f.create_dataset('fens', data=fens[selected_indices])
    f.create_dataset('k', data=ks[selected_indices])

print("Subset dataset created successfully!")
