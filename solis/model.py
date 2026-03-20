"""
SOLIS: Chess engine that plays entirely within a learned contrastive embedding space.

The model encodes chess positions (as 77-token FEN sequences) into an embedding space
where positions with similar win probabilities are clustered together. The resulting
manifold is linear: one end corresponds to white checkmate, the other to black checkmate,
with equal positions near the center.

Architecture: Token Embedding -> Learned Positional Encoding -> [CLS] + Transformer Encoder -> Linear Projection -> L2 Norm

Training uses supervised contrastive loss (SupConLoss) where positions within a
win-probability threshold of each other are treated as positives.
"""

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from solis.loss import SupConLoss


class SOLIS(L.LightningModule):
    """
    Transformer-based chess position encoder trained with supervised contrastive learning.

    Encodes 77-token FEN representations into L2-normalized embeddings. During inference,
    moves are selected by projecting resulting position embeddings onto an "advantage
    direction" (the vector from mean black-checkmate embedding to mean white-checkmate
    embedding) and using beam search to find the best trajectory.

    Args:
        embed_dim: Dimension of token embeddings and output space.
        num_heads: Number of attention heads in each transformer layer.
        ff_dim: Hidden dimension of the feedforward network in each transformer layer.
        num_layers: Number of stacked transformer encoder layers.
        lr: Learning rate for SGD optimizer.
        momentum: Momentum for SGD optimizer.
        num_positives: Number of positive samples per anchor during training.
        p_threshold: Win-probability distance threshold for defining positive pairs.
    """

    def __init__(self, embed_dim=512, num_heads=16, ff_dim=512, num_layers=6,
                 lr=0.05, momentum=0.9, num_positives=5, p_threshold=0.05):
        super().__init__()

        self.save_hyperparameters()

        self.seq_len = 77  # fixed tokenized FEN length

        # Token embedding maps each of the 34 possible characters to embed_dim
        self.embedding = nn.Embedding(self.seq_len, embed_dim)
        # Learned positional encoding (+1 for prepended CLS token)
        self.positional_encoding = nn.Parameter(torch.randn(1, self.seq_len + 1, embed_dim))
        # CLS token aggregates sequence information (similar to BERT)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Project to embedding space (same dimensionality, but learned linear transform)
        self.projection = nn.Linear(embed_dim, embed_dim)

        self.loss_fn = SupConLoss()

    def forward(self, x):
        """
        Encode tokenized FEN positions into L2-normalized embeddings.

        Args:
            x: Integer tensor of shape [batch_size, 77] containing tokenized FEN positions.

        Returns:
            L2-normalized embeddings of shape [batch_size, embed_dim].
        """
        x = x.long()
        batch_size, seq_len = x.shape

        x = self.embedding(x)

        # Prepend [CLS] token, then add learned positional encoding
        cls_tokens = self.cls_token.repeat(batch_size, 1, 1)
        x = torch.cat([cls_tokens, x], dim=1)
        e = self.positional_encoding[:, : x.shape[1], :]
        x = x + e

        x = self.transformer(x)

        # Extract CLS token representation -> project -> L2 normalize
        x = x[:, 0]
        x = self.projection(x)
        return F.normalize(x, p=2, dim=-1)

    def training_step(self, batch, batch_idx):
        anchor_token = batch["anchor"]
        positive_tokens = batch["positives"]
        ps = batch["label"]

        b = ps.shape[0]

        # anchor embeddings
        ae = self(anchor_token)

        # positive embeddings
        pe = self(positive_tokens.view(-1, self.seq_len))
        pe = pe.view(b, self.hparams.num_positives, -1)

        # combine
        embeddings = torch.cat([ae.unsqueeze(1), pe], dim=1)

        with torch.no_grad():
            ps = ps.view(-1, 1)
            p_diffs = torch.abs(ps - ps.T)
            mask = (p_diffs < self.hparams.p_threshold).float()

        loss = self.loss_fn(embeddings, labels=None, mask=mask)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        anchor_token = batch["anchor"]
        positive_tokens = batch["positives"]
        ps = batch["label"]

        b = ps.shape[0]

        ae = self(anchor_token)
        pe = self(positive_tokens.view(-1, self.seq_len))
        pe = pe.view(b, self.hparams.num_positives, -1)

        embeddings = torch.cat([ae.unsqueeze(1), pe], dim=1)

        # p-distance mask
        with torch.no_grad():
            ps = ps.view(-1, 1)
            p_diffs = torch.abs(ps - ps.T)
            mask = (p_diffs < self.hparams.p_threshold).float()

        loss = self.loss_fn(embeddings, labels=None, mask=mask)

        self.log("val_loss",
                 loss,
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True,
                 sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum
        )
        return optimizer
