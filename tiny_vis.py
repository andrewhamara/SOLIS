import torch
import h5py
import numpy as np
import umap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from tokenizer import tokenize
from solis import SOLIS
import chess
import chess.pgn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# load model
#MODEL_PATH = "/data/hamaraa/solis_latest_small.pth"
MODEL_PATH = "/data/hamaraa/solis_tiny_stepstep=370000.ckpt"
#MODEL_PATH = "/data/hamaraa/solis_good.pth"
#model = SOLIS().to(DEVICE)
#model = SOLIS(embed_dim=1024, ff_dim=1024).to(DEVICE)
model = SOLIS(embed_dim=256, ff_dim=512, num_heads=8, num_layers=6).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)["state_dict"])
#model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

#v_white = np.load("/data/hamaraa/mean_white_checkmate.npy")
#v_black = np.load("/data/hamaraa/mean_black_checkmate.npy")

# load data
DATA_PATH = "/data/hamaraa/tokenized_1m.h5"  # or tokenized_1m.h5 if you dropped bins
print("Loading data...")
with h5py.File(DATA_PATH, "r") as f:
    tokens = np.array(f["fens"])
    ps = np.array(f["ps"])

# subsample data
N = 100_000
indices = np.random.choice(len(tokens), size=N, replace=False)
tokens_subset = tokens[indices]
ps_subset = ps[indices]

# encode samples
def get_embeddings(x, batch_size=256):
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch = torch.tensor(x[i:i+batch_size], dtype=torch.long, device=DEVICE)
            emb = model(batch).cpu().numpy()
            all_embeddings.append(emb)
    return np.concatenate(all_embeddings, axis=0)

print("Running model inference...")
embeddings = get_embeddings(tokens_subset)

# UMAP
print("Running UMAP...")
reducer = umap.UMAP(n_components=2, random_state=42)
embeddings_2d = reducer.fit_transform(embeddings)

#v_white_2d = reducer.transform(v_white.reshape(1, -1))[0]
#v_black_2d = reducer.transform(v_black.reshape(1, -1))[0]

# plot: color only 1.0, 0.5, 0.0
plt.figure(figsize=(10, 8))

scatter = plt.scatter(
    embeddings_2d[:, 0],
    embeddings_2d[:, 1],
    c=ps_subset,
    cmap="coolwarm",  # blue = black win, red = white win
    s=8,
    alpha=0.8
)


# minimal legend, no colorbar
plt.tight_layout()
plt.savefig("solis_umap_winprob.pdf")
plt.close()
