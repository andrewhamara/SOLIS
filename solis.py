import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import SupConLoss


class SOLIS(L.LightningModule):
    def __init__(self, embed_dim=512, num_heads=16, ff_dim=512, num_layers=6,
                 lr=0.05, momentum=0.9, num_positives=5, p_threshold=0.05):
        super().__init__()

        self.save_hyperparameters()

        # token sequence length
        self.seq_len = 77

        # model architecture
        self.embedding = nn.Embedding(self.seq_len, embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, self.seq_len + 1, embed_dim))
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
        self.projection = nn.Linear(embed_dim, embed_dim)

        self.loss_fn = SupConLoss()

    def forward(self, x):
        x = x.long()
        batch_size, seq_len = x.shape

        x = self.embedding(x)

        # prepend special [CLS] token
        cls_tokens = self.cls_token.repeat(batch_size, 1, 1)
        x = torch.cat([cls_tokens, x], dim=1)

        # add positional encoding (learned)
        e = self.positional_encoding[:, : x.shape[1], :]
        x = x + e 

        # forward pass
        x = self.transformer(x)

        # extract CLS token
        x = x[:, 0]

        # projection into embedding space followed by norm
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

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum
        )
        return optimizer