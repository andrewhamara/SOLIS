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
MODEL_PATH = "/data/hamaraa/solis_latest_small.pth"
MODEL_PATH = "/data/hamaraa/solis_good.pth"
#model = SOLIS().to(DEVICE)
model = SOLIS(embed_dim=1024, ff_dim=1024).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

v_white = np.load("/data/hamaraa/mean_white_checkmate.npy")
v_black = np.load("/data/hamaraa/mean_black_checkmate.npy")

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

v_white_2d = reducer.transform(v_white.reshape(1, -1))[0]
v_black_2d = reducer.transform(v_black.reshape(1, -1))[0]

# plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    embeddings_2d[:, 0],
    embeddings_2d[:, 1],
    c=ps_subset,
    cmap="coolwarm",  # blue = black win, red = white win
    s=2,
    alpha=0.6,
    linewidth=0,
    rasterized=True
)
plt.annotate("",
    xy=v_white_2d, xytext=v_black_2d,
    arrowprops=dict(arrowstyle="->", color="black", lw=2, linestyle="--"),
)

cbar = plt.colorbar(scatter, shrink=0.8, pad=0.01)
cbar.set_label("Win Probability (for white)")

plt.axis("off")
plt.tight_layout()
plt.savefig("solis_umap_colored.pdf")
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
    ax.scatter(base_points[:, 0], base_points[:, 1], s=1, alpha=0.05, color="gray")

    # nodes
    ax.scatter(traj_points_2d[1:-1, 0], traj_points_2d[1:-1, 1], color="black", s=10)
    #ax.scatter(traj_points_2d[0, 0], traj_points_2d[0, 1], color="green", s=60, label="start")
    #ax.scatter(traj_points_2d[-1, 0], traj_points_2d[-1, 1], color="black", s=60, label="end")

    # arrows for each move
    for i in range(len(traj_points_2d) - 1):
        x1, y1 = traj_points_2d[i]
        x2, y2 = traj_points_2d[i + 1]
        ax.annotate("",
            xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5, alpha=0.8),
        )

    #ax.set_title("Latent Interpolation of Game Trajectory")
    ax.axis("off")
    #ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


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
PGN_PATHS = ["newgame.pgn", "drawgame.pgn", "blackgame.pgn"]

for PGN_PATH in PGN_PATHS:
    game = PGN_PATH.split('.')[0]
    game_fens = extract_fens_from_pgn(PGN_PATH)
    print("Embedding trajectory...")
    z_traj = encode_fens(game_fens)
    z_traj_2d = reducer.transform(z_traj)
    plot_game_on_umap(embeddings_2d, z_traj_2d, f"solis_{game}_interpolation.pdf")
