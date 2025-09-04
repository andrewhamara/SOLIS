import torch
import h5py
import numpy as np
from solis import SOLIS
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# load model
model = SOLIS(embed_dim=1024, ff_dim=1024).to(DEVICE)
model.load_state_dict(torch.load("/data/hamaraa/solis_good.pth", map_location=DEVICE))
model.eval()

# load data
DATA_PATH = "/data/hamaraa/tokenized_1m.h5"
with h5py.File(DATA_PATH, 'r') as f:
    tokens = np.array(f["fens"])
    ps = np.array(f["ps"])

# filter
white_idxs = np.where(ps == 1.0)[0]
black_idxs = np.where(ps == 0.0)[0]

print(f"White mate positions: {len(white_idxs)}")
print(f"Black mate positions: {len(black_idxs)}")

def batch_embeddings(indices, batch_size=256):
    embs = []
    for i in tqdm(range(0, len(indices), batch_size), desc="Encoding"):
        batch = torch.tensor(tokens[indices[i:i+batch_size]], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            emb = model(batch).cpu().numpy()
        embs.append(emb)
    return np.concatenate(embs, axis=0)

# get mean vectors
white_embs = batch_embeddings(white_idxs)
black_embs = batch_embeddings(black_idxs)
mean_white = white_embs.mean(axis=0)
mean_black = black_embs.mean(axis=0)

# save
np.save("/data/hamaraa/mean_white_checkmate.npy", mean_white)
np.save("/data/hamaraa/mean_black_checkmate.npy", mean_black)

print('done')
