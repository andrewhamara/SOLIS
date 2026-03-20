import h5py

# path to your original un-tokenized file
dataset_path = "/data/hamaraa/solis_subset_10k.h5"

# open and inspect
with h5py.File(dataset_path, "r") as f:
    fens = f["fens"]
    ps = f["ps"]

    print(f"Total samples: {len(fens)}\n")

    for i in range(5):
        fen = fens[i].decode("utf-8")
        p = ps[i]
        print(f"Sample {i}:")
        print(f"  FEN: {fen}")
        print(f"  Win Probability: {p:.4f}")
        print()
