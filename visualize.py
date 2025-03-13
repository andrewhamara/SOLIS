import torch
import h5py
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from solis import SOLIS  # Import the trained model
from tokenizer import tokenize  # Your custom tokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/data/hamaraa/solis_epoch_300.pth"  # Adjust this to your trained model checkpoint
model = SOLIS().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print('loading dataset...')
DATA_PATH = "/data/hamaraa/mate_in_k_test.h5"
with h5py.File(DATA_PATH, "r") as f:
    fens = np.array(f["fens"])
    ks = np.array(f["k"])

print('extracting checkmates...')
mate_white_indices = ks == 444
mate_black_indices = ks == -444
normal_indices = (ks != 444) & (ks != -444)

print('doing forward passes...')
def inference(fens):
    with torch.no_grad():
        fen_tokens = torch.tensor([tokenize(fen.decode('utf-8')) for fen in fens]).to(DEVICE)
        embeddings = model(fen_tokens).cpu().numpy()
        return embeddings

normal_fens = fens[normal_indices]
normal_ks = ks[normal_indices]
normal_embeddings = inference(normal_fens)

white_mate_fens = fens[mate_white_indices]
white_mate_embeddings = inference(white_mate_fens)

black_mate_fens = fens[mate_black_indices]
black_mate_embeddings = inference(black_mate_fens)

print(len(normal_fens))
print(len(white_mate_fens))
print(len(black_mate_fens))

print('reducing dimensions...')
umap_reducer = umap.UMAP(n_components=2, random_state=42)
normal_embeddings_2d = umap_reducer.fit_transform(normal_embeddings)
white_mate_embeddings_2d = umap_reducer.transform(white_mate_embeddings)
black_mate_embeddings_2d = umap_reducer.transform(black_mate_embeddings)

unique_ks = np.unique(normal_ks)
color_palette = sns.color_palette("tab20", len(unique_ks))  # Use tab20 for categorical colors
k_to_color = {k: color_palette[i] for i, k in enumerate(unique_ks)}

# Map k values to colors
point_colors = [k_to_color[k] for k in normal_ks]

# Plot UMAP embeddings
plt.figure(figsize=(12, 8))

# Plot normal positions (dots)
plt.scatter(
    normal_embeddings_2d[:, 0],
    normal_embeddings_2d[:, 1],
    c=point_colors,
    alpha=0.7,
    s=5,
    label="Mate in k positions"
)

# Plot white checkmate positions (star markers)
plt.scatter(
    white_mate_embeddings_2d[:, 0],
    white_mate_embeddings_2d[:, 1],
    c="gold",
    marker="*",  # Star marker
    s=80,        # Bigger size to make it distinct
    edgecolors="black",
    label="White Checkmate (k=+444)"
)

# Plot black checkmate positions (star markers)
plt.scatter(
    black_mate_embeddings_2d[:, 0],
    black_mate_embeddings_2d[:, 1],
    c="red",
    marker="*",
    s=80,
    edgecolors="black",
    label="Black Checkmate (k=-444)"
)

# Add legend for k values
for k, color in k_to_color.items():
    plt.scatter([], [], c=[color], label=f'k={k}')

plt.legend(markerscale=3, title="Mate in k")
plt.title("UMAP Projection of Chess Positions (Colored by k, Checkmates as Stars)")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.savefig('embeddings.pdf')
