import random
from bagz import BagDataSource
from constants import CODERS
import chess
import numpy as np
import h5py

data_path = "/data/hamaraa/chess_data.bag"
output_path = "/data/hamaraa/solis_test_5m.h5"

# open bag file
source = BagDataSource(data_path)
n_total = len(source)
n_keep = 5_000_000

# sample
keep_indices = sorted(random.sample(range(n_total), n_keep))

fens, ps = [], []

for i, idx in enumerate(keep_indices):
    record = source[idx]
    fen, p = CODERS['state_value'].decode(record)
    
    b = chess.Board(fen)

    # convert to probability for white
    if b.turn == chess.BLACK:
        p = 1.0 - p

    fens.append(fen)
    ps.append(p)

    if i % 10000 == 0:
        print(f"{i}/{n_keep}")

with h5py.File(output_path, 'w') as f:
    f.create_dataset('fens', data=np.array(fens, dtype='S'))
    f.create_dataset('ps', data=np.array(ps, dtype=np.float32))
