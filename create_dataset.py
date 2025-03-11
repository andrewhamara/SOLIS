import h5py
import concurrent.futures
import chess
import numpy as np

import utils
import constants
import bagz
from mate import get_k


# paths
DATA_PATH = '/data/hamaraa/chess_data.bag'
OUTPUT_H5_PATH = '/data/hamaraa/mate_in_k_dataset.h5'


data_source = bagz.BagDataSource(DATA_PATH)


# print number of records
num_records = len(data_source)
print(f'records: {num_records}')

def process_batch(start_idx, end_idx):

    fens, ks = [], []

    for i in range(start_idx, end_idx):
        raw_record = data_source[i]
        fen, win = constants.CODERS['state_value'].decode(raw_record)
        
        if win in {0.0, 1.0}:
            k = get_k(fen)
            if k is not None:
                fens.append(fen)
                ks.append(k)

    pct = end_idx / num_records
    print(pct)
    return fens, ks

BATCH_SIZE = 10000
NUM_WORKERS = 48

all_fens, all_ks = [], []
with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = [
        executor.submit(process_batch, start, min(start + BATCH_SIZE, num_records))
        for start in range(0, num_records, BATCH_SIZE)
    ]

    for future in concurrent.futures.as_completed(futures):
        fens, ks = future.result()
        all_fens.extend(fens)
        all_ks.extend(ks)

with h5py.File(OUTPUT_H5_PATH, 'w') as f:
    f.create_dataset('fens', data=np.array(all_fens, dtype='S'))
    f.create_dataset('k', data=np.array(all_ks, dtype=np.int32))

print('data saved')
