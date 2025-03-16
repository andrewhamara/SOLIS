import torch
import h5py
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from solis import SOLIS  # Import the trained model
from tokenizer import tokenize  # Your custom tokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
MODEL_PATH = "/data/hamaraa/solis_best.pth"  # Adjust this to your trained model checkpoint
#model = SOLIS(embed_dim=64, ff_dim=512, num_heads=8, num_layers=6).to(DEVICE)
model = SOLIS().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print('loading dataset...')
DATA_PATH = "/data/hamaraa/mate_in_k_train_100k.h5"
with h5py.File(DATA_PATH, "r") as f:
    fens = np.array(f["fens"])
    ks = np.array(f["k"])

print(len(fens))
print(len(ks))

print('extracting checkmates...')
mate_white_indices = ks == 444
mate_black_indices = ks == -444
normal_indices = (ks != 444) & (ks != -444)

print('doing forward passes...')
def inference(fens_batch, batch_size=256):
    with torch.no_grad():
        all_embeddings = []
        for i in range(0, len(fens_batch), batch_size):
            batch = torch.tensor(fens_batch[i:i+batch_size], dtype=torch.long, device=DEVICE)
            batch_embeddings = model(batch).cpu().numpy()
            all_embeddings.append(batch_embeddings)
        return np.concatenate(all_embeddings, axis=0)

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

plt.figure(figsize=(12, 8))

# Plot normal positions (dots)
scatter = plt.scatter(
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

# Add dummy scatter points for each k value
for k, color in k_to_color.items():
    plt.scatter([], [], c=[color], label=f'k={k}')

# Create a proper legend
plt.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.05, 1), 
           borderaxespad=0., markerscale=3, title="Mate in k")

# Ensure that the legend fits within the figure
plt.subplots_adjust(right=0.75)  # Adjust right margin to fit legend

plt.title("UMAP Projection of Chess Positions (Colored by k, Checkmates as Stars)")
plt.savefig('embeddings.pdf')
