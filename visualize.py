import torch
import h5py
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from tokenizer import tokenize
from solis import SOLIS
import chess
import chess.pgn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# load model
MODEL_PATH = "/data/hamaraa/solis_good.pth"
#model = SOLIS().to(DEVICE)
model = SOLIS(embed_dim=1024, ff_dim=1024).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# load data
DATA_PATH = "/data/hamaraa/tokenized_1m.h5"  # or tokenized_1m.h5 if you dropped bins
print("Loading data...")
with h5py.File(DATA_PATH, "r") as f:
    tokens = np.array(f["fens"])
    ps = np.array(f["ps"])

# subsample data
N = 10_000
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

# plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    embeddings_2d[:, 0],
    embeddings_2d[:, 1],
    c=ps_subset,
    cmap="coolwarm",  # blue = black win, red = white win
    s=8,
    alpha=0.8
)
cbar = plt.colorbar(scatter)
cbar.set_label("Win Probability (for white)")
plt.title("UMAP Projection of SOLIS Embeddings\nColored by Win Probability")
plt.tight_layout()
plt.savefig("solis_umap_winprob.pdf")
plt.close()

# === optional: overlay game trajectory ===
def encode_fens(fens):
    model.eval()
    with torch.no_grad():
        toks = [tokenize(fen) for fen in fens]
        toks = torch.tensor(toks, dtype=torch.long, device=DEVICE)
        return model(toks).cpu().numpy()

def plot_game_on_umap(base_points, traj_points_2d, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))

    # background cloud
    ax.scatter(base_points[:, 0], base_points[:, 1], s=1, alpha=0.05, color="gray", label="embedding space")

    # nodes
    ax.scatter(traj_points_2d[1:-1, 0], traj_points_2d[1:-1, 1], color="black", s=10)
    ax.scatter(traj_points_2d[0, 0], traj_points_2d[0, 1], color="green", s=60, label="start")
    ax.scatter(traj_points_2d[-1, 0], traj_points_2d[-1, 1], color="black", s=60, label="end")

    # arrows for each move
    for i in range(len(traj_points_2d) - 1):
        x1, y1 = traj_points_2d[i]
        x2, y2 = traj_points_2d[i + 1]
        ax.annotate("",
            xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="black", lw=1),
        )

    ax.set_title("Latent Interpolation of Game Trajectory")
    ax.axis("off")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# example 4 move checkmate
def extract_fens_from_pgn(pgn_path):
    with open(pgn_path) as f:
        game = chess.pgn.read_game(f)
    board = game.board()
    fens = [board.fen()]
    for move in game.mainline_moves():
        board.push(move)
        fens.append(board.fen())
    return fens

# === Inputs ===
PGN_PATH = "game.pgn"
game_fens = extract_fens_from_pgn(PGN_PATH)

# === run trajectory visualization ===
print("Embedding trajectory...")
z_traj = encode_fens(game_fens)
z_traj_2d = reducer.transform(z_traj)
plot_game_on_umap(embeddings_2d, z_traj_2d, "solis_game_interpolation.pdf")
