from lightning.pytorch import Trainer
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint
from solis import SOLIS
from dataloader import get_dataloader
from lightning.pytorch.loggers import CSVLogger

# === Hyperparameters ===
BATCH_SIZE = 128
NUM_POSITIVES = 5
MAX_STEPS = 500_000
GPUS = 4

# === Dataloader ===
train_dataloader = get_dataloader(
    batch_size=BATCH_SIZE,
    split='train',
    k_pos=NUM_POSITIVES
)

val_dataloader = get_dataloader(
    batch_size=BATCH_SIZE,
    split='val',
    k_pos=NUM_POSITIVES
)


# === Model ===
solis_small = SOLIS(
    embed_dim=512,
    num_heads=16,
    ff_dim=512,
    num_layers=6,
    lr=0.05,
    momentum=0.9,
    num_positives=NUM_POSITIVES,
    p_threshold=0.05
)

# === Checkpointing (every 20k steps) ===
checkpoint_callback = ModelCheckpoint(
    dirpath="/data/hamaraa/",
    filename="solis_small_step{step}",
    every_n_train_steps=10_000,
    save_top_k=-1,  # save all checkpoints
    save_weights_only=True
)

best_ckpt = ModelCheckpoint(
    dirpath="/data/hamaraa/",
    filename="solis_small_best",
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    save_weights_only=True
)

csv_logger = CSVLogger("logs", name="solis")

# === Trainer ===
trainer = Trainer(
    accelerator="gpu",
    devices=GPUS,
    strategy=DDPStrategy(find_unused_parameters=False),
    max_steps=MAX_STEPS,
    log_every_n_steps=10,
    logger=csv_logger,
    callbacks=[checkpoint_callback, best_ckpt]
)

# === Train ===
trainer.fit(solis_small, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# === Final Save ===
model_path = "/data/hamaraa/solis_final_small.ckpt"
trainer.save_checkpoint(model_path)
print(f"Model saved to {model_path}")