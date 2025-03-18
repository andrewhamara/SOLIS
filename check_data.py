import h5py
import numpy as np
import torch
from mate_in_k_dataloader import get_dataloader
import random

# Path to your dataset
DATA_PATH = "/data/hamaraa/mate_in_k_dataset.h5"

def check_h5_file():
    """ Verify that the .h5 file loads correctly and contains expected keys """
    with h5py.File(DATA_PATH, "r") as f:
        print("Keys in dataset:", list(f.keys()))

        if "fens" not in f or "k" not in f:
            print("Error: Missing required keys ('fens' or 'k') in dataset!")
            return False

        fens = np.array(f["fens"])
        ks = np.array(f["k"])

        print(f"Total positions: {len(fens)}")
        print(f"Unique k values: {np.unique(ks)}")

        # Check if k values are in expected range
        if not np.all((-444 <= ks) & (ks <= 444)):
            print("Error: Found unexpected k values outside the expected range (-444 to 444).")
            return False

    return True

def verify_sample_positions(num_samples=10):
    """ Extract a few samples and verify the FENs are correctly associated with k values """
    with h5py.File(DATA_PATH, "r") as f:
        fens = np.array(f["fens"])
        ks = np.array(f["k"])

        for _ in range(num_samples):
            idx = random.randint(0, len(fens) - 1)
            print(f"Sample {idx}: FEN = {fens[idx]}, k = {ks[idx]}")

def check_duplicates():
    """ Check for duplicate positions in the dataset """
    with h5py.File(DATA_PATH, "r") as f:
        fens = np.array(f["fens"])
        unique_fens = len(np.unique(fens))
        print(f"Total FENs: {len(fens)}, Unique FENs: {unique_fens}")

        if unique_fens != len(fens):
            print("Warning: There are duplicate FENs in the dataset.")

def check_dataloader_integrity(batch_size=32):
    """ Load samples from the dataloader and verify they align with k values """
    dataloader = get_dataloader(batch_size=batch_size, split='train')
    ks = dataloader.dataset.ks
    fens = dataloader.dataset.fens

    for batch in dataloader:
        anchors, positives, negatives = batch

        print(f"Batch size: {len(anchors)}")
        print(f"Anchor FEN (sample): {anchors[0]}")
        print(f"Positive FEN (sample): {positives[0]}")
        print(f"Negative FEN (sample): {negatives[0]}")

        # Confirm the anchor and positive have the same k, while negatives do not
        anchor_k = ks[np.where(fens == anchors[0])]
        positive_k = ks[np.where(fens == positives[0])]
        negative_ks = [ks[np.where(fens == neg)] for neg in negatives]

        print(f"Anchor k: {anchor_k}, Positive k: {positive_k}, Negative ks: {negative_ks}")

        if anchor_k != positive_k:
            print("Error: Positive example does not match anchor in k!")

        if any(anchor_k == neg_k for neg_k in negative_ks):
            print("Error: Found negative samples with the same k as the anchor!")

        break  # Only check the first batch

if __name__ == "__main__":
    print("Checking dataset integrity...")
    if check_h5_file():
        #print("\nVerifying sample positions...")
        #verify_sample_positions()

        print("\nChecking for duplicates...")
        check_duplicates()

        #print("\nVerifying dataloader integrity...")
        #check_dataloader_integrity()
