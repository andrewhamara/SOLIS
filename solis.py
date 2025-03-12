import torch
import torch.nn as nn
import torch.nn.functional as F


class SOLIS(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, ff_dim=2048, num_layers=6):
        super().__init__()

        self.embedding = nn.Linear(77, embed_dim)
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

    def forward(self, x):
        x = x.float()
        x = self.embedding(x)
        x = self.transformer(x.unsqueeze(1))
        x = self.projection(x.squeeze(1))
        return F.normalize(x, p=2, dim=-1)
