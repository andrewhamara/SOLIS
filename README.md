# SOLIS

SOLIS learns a manifold where chess positions are organized by game outcome: one end corresponds to white checkmate, the other to black checkmate, and equal positions lie near the center. Moves are selected by finding actions that push the position toward favorable regions of this space using beam search along a learned "advantage direction."

<p align="center">
  <img src="orbit.gif" alt="3D orbit of the SOLIS embedding space" width="500"/>
  <br>
  <em>3D projection of the learned embedding space. The structure from black checkmate to white checkmate emerges naturally from contrastive training.</em>
</p>

## Papers

- **Learning to Plan via Supervised Contrastive Learning and Strategic Interpolation: A Chess Case Study** — [arXiv:2506.04892](https://arxiv.org/abs/2506.04892)
- **Latent Planning via Embedding Arithmetic: A Contrastive Approach to Strategic Reasoning** — [arXiv:2511.09477](https://arxiv.org/abs/2511.09477)

## How It Works

### Training

SOLIS uses **supervised contrastive learning** to train a Transformer encoder on millions of chess positions. Positions with similar win probabilities (within a threshold `p_threshold`) are pulled together in embedding space, while dissimilar positions are pushed apart.

```
FEN string → Tokenizer (77 tokens) → Transformer Encoder → L2-normalized embedding (1024-d)
```

The result is a manifold with that encodes value as distance.

### Inference

At inference time, SOLIS computes an **advantage direction**, the normalized vector from the mean black-checkmate embedding to the mean white-checkmate embedding. Positions are scored by projecting their embeddings onto this axis:

```
score(position) = (embedding − μ_black) · (μ_white − μ_black)
```

A greedy search explores moves by expanding only the top-k most promising moves at each depth level, based on scores given by the advantage axis.

## Setup

### Dependencies

```bash
pip install torch lightning python-chess numpy h5py tqdm matplotlib jaxtyping
```

You'll also need [Stockfish](https://stockfishchess.org/) installed for evaluation. Pass its path via the `--stockfish` flag.

### Data

Training data should be pre-tokenized into HDF5 format with two datasets:
- `fens`: tokenized FEN positions (uint8 arrays of length 77)
- `ps`: corresponding win probabilities (float)

Use `scripts/tokenize_dataset.py` to convert raw FEN data, and `data/split_dataset.py` to create train/val splits.

## Usage

### Training

```bash
python train.py --data_path /path/to/tokenized.h5 --checkpoint_dir checkpoints/
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--embed_dim` | 512 | Embedding dimension |
| `--ff_dim` | 512 | Feedforward dimension |
| `--num_heads` | 16 | Attention heads |
| `--num_layers` | 6 | Transformer layers |
| `--batch_size` | 128 | Samples per GPU |
| `--num_positives` | 5 | Positive pairs per anchor |
| `--max_steps` | 500,000 | Total training steps |
| `--gpus` | 4 | Number of GPUs |
| `--p_threshold` | 0.05 | Win-probability threshold for positive pairs |

### Evaluation against Stockfish

```bash
python evaluate.py \
    --checkpoint /path/to/model.pth \
    --v_white /path/to/mean_white_checkmate.npy \
    --v_black /path/to/mean_black_checkmate.npy \
    --stockfish /path/to/stockfish \
    --depth 4 --width 3 --elo 2200 --games 100
```

Games are saved as PGN files. The engine alternates between playing as white and black.

## Project Structure

```
solis/                   Core library (importable package)
├── model.py             SOLIS model — Transformer encoder with contrastive training
├── tokenizer.py         FEN → 77-token integer vector
├── loss.py              Supervised Contrastive Loss (SupConLoss)
├── dataloader.py        HDF5 dataset loader with positive-pair sampling
└── search.py            Latent beam search engine with LRU caching

train.py                 Training entry point (PyTorch Lightning + DDP)
evaluate.py              Unified Stockfish evaluation with CLI args

scripts/                 Data processing and experiment utilities
├── create_dataset.py    Create HDF5 datasets from raw data
├── tokenize_dataset.py  Tokenize FEN strings into integer vectors
├── get_mate_embeddings.py  Compute mean checkmate embeddings (v_white, v_black)
├── launch_sweep.py      Launch hyperparameter sweep across GPUs
├── queue.sh             Batch job launcher for evaluation sweeps
└── ...                  Additional data processing utilities

visualization/           Embedding space visualization
├── visualize_2d.py      2D UMAP projection with game trajectories
└── visualize_3d.py      3D UMAP orbit animation

data/                    Dataset creation and preprocessing
experiments/             Legacy per-config evaluation scripts (for reference)
```

## Model Variants

| Variant | `embed_dim` | `ff_dim` | Notes |
|---------|------------|----------|-------|
| **Base** | 1024 | 1024 | Strongest; used in paper evaluations |
| **Small** | 512 | 512 | Default in `train.py` |
| **Mini** | 128 | 256 | Lightweight for fast experiments |
| **Tiny** | 256 | 512 | Minimal variant |

## Citation

```bibtex
@misc{hamara2025learningplansupervisedcontrastive,
      title={Learning to Plan via Supervised Contrastive Learning and Strategic Interpolation: A Chess Case Study},
      author={Andrew Hamara and Greg Hamerly and Pablo Rivas and Andrew C. Freeman},
      year={2025},
      eprint={2506.04892},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.04892},
}

@misc{hamara2025latentplanningembeddingarithmetic,
      title={Latent Planning via Embedding Arithmetic: A Contrastive Approach to Strategic Reasoning},
      author={Andrew Hamara and Greg Hamerly and Pablo Rivas and Andrew C. Freeman},
      year={2025},
      eprint={2511.09477},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.09477},
}
```

## License

Please see the associated papers for usage terms.
