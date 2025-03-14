import h5py
import numpy as np
import torch
from tokenizer import tokenize  # Your existing tokenizer

DATA_PATH = '/data/hamaraa/mate_in_k_train.h5'
TOKENIZED_DATA_PATH = '/data/hamaraa/mate_in_k_train_tokenized.h5'

print("Loading raw FEN dataset...")
with h5py.File(DATA_PATH, 'r') as f:
    fens = np.array(f['fens'])
    ks = np.array(f['k'])

print(f"Tokenizing {len(fens)} FENs...")
tokenized_fens = np.array([tokenize(fen.decode('utf-8')) for fen in fens], dtype=np.int32)

print("Saving tokenized dataset...")
with h5py.File(TOKENIZED_DATA_PATH, 'w') as f:
    f.create_dataset('fens', data=tokenized_fens)
    f.create_dataset('k', data=ks)

print("Tokenization complete! Saved to", TOKENIZED_DATA_PATH)
