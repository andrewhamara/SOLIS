"""
Training script for SOLIS.

Trains a SOLIS model using supervised contrastive learning with multi-GPU DDP.
Checkpoints are saved periodically and the best model is tracked by validation loss.

Usage:
    python train.py --data_path /path/to/tokenized.h5 [--embed_dim 512] [--gpus 4]
"""

import argparse

from lightning.pytorch import Trainer
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from solis.model import SOLIS
from solis.dataloader import get_dataloader


def main():
    parser = argparse.ArgumentParser(description="Train SOLIS")
    parser.add_argument("--data_path", type=str, required=True, help="Path to tokenized HDF5 dataset")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/", help="Directory to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_positives", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=500_000)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--ff_dim", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--p_threshold", type=float, default=0.05)
    args = parser.parse_args()

    # Dataloaders
    train_dataloader = get_dataloader(
        h5_path=args.data_path,
        batch_size=args.batch_size,
        split='train',
        k_pos=args.num_positives,
    )
    val_dataloader = get_dataloader(
        h5_path=args.data_path,
        batch_size=args.batch_size,
        split='val',
        k_pos=args.num_positives,
    )

    # Model
    model = SOLIS(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        num_layers=args.num_layers,
        lr=args.lr,
        momentum=0.9,
        num_positives=args.num_positives,
        p_threshold=args.p_threshold,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="solis_step{step}",
        every_n_train_steps=10_000,
        save_top_k=-1,
        save_weights_only=True,
    )
    best_ckpt = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="solis_best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_weights_only=True,
    )

    csv_logger = CSVLogger("logs", name="solis")

    # Trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=args.gpus,
        strategy=DDPStrategy(find_unused_parameters=False),
        max_steps=args.max_steps,
        log_every_n_steps=10,
        logger=csv_logger,
        callbacks=[checkpoint_callback, best_ckpt],
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # Final save
    final_path = f"{args.checkpoint_dir}/solis_final.ckpt"
    trainer.save_checkpoint(final_path)
    print(f"Model saved to {final_path}")


if __name__ == "__main__":
    main()
