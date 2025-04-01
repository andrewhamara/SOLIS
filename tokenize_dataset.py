import h5py
import numpy as np
import torch
from tokenizer import tokenize

DATA_PATH = '/data/hamaraa/solis_test_5m.h5'
TOKENIZED_DATA_PATH = '/data/hamaraa/tokenized_5m.h5'

print("Loading raw FEN dataset...")
with h5py.File(DATA_PATH, 'r') as f:
    fens = np.array(f['fens'])
    ps = np.array(f['ps'])

print(f"Tokenizing {len(fens)} FENs...")
tokenized_fens = np.array([tokenize(fen.decode('utf-8')) for fen in fens], dtype=np.int32)

print("Saving tokenized dataset...")
with h5py.File(TOKENIZED_DATA_PATH, 'w') as f:
    f.create_dataset('fens', data=tokenized_fens)
    f.create_dataset('ps', data=ps)

print("Tokenization complete! Saved to", TOKENIZED_DATA_PATH)
