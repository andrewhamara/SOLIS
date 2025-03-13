import h5py
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Path to your h5 file
DATA_PATH = '/data/hamaraa/mate_in_k_dataset.h5'

# Load k values from h5 file
with h5py.File(DATA_PATH, 'r') as f:
    ks = np.array(f['k'])

print(len(ks))
# Count occurrences of each k
k_counts = Counter(ks)

# Sort keys for proper visualization
sorted_ks = sorted(k_counts.keys())
#sorted_ks.remove(-444)
#sorted_ks.remove(444)
sorted_counts = [k_counts[k] for k in sorted_ks]

# Print out the distribution
print("k Distribution:")
for k, count in zip(sorted_ks, sorted_counts):
    with open('ks.txt', 'a+') as f:
        f.write(f"k={k}: {count} positions\n")


# Plot distribution
plt.figure(figsize=(10, 5))
plt.bar(sorted_ks, sorted_counts, width=0.8, color='blue', alpha=0.7)
plt.xlabel("k value")
plt.ylabel("Frequency")
plt.title("Distribution of k values in dataset")
plt.xticks(rotation=90)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig('dist.pdf')
