import torch
import torch.nn as nn
import torch.nn.functional as F


class SOLIS(nn.Module):
    def __init__(self, embed_dim=2048, num_heads=16, ff_dim=2048, num_layers=6):
        super().__init__()

        self.embed_dim = embed_dim
        self.seq_len = 77

        self.embedding = nn.Embedding(self.seq_len, embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, self.seq_len + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.0,
            activation='gelu',
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        x = x.long()
        batch_size, seq_len = x.shape

        x = self.embedding(x)

        cls_tokens = self.cls_token.repeat(batch_size, 1, 1)
        x = torch.cat([cls_tokens, x], dim=1)

        e = self.positional_encoding[:, : x.shape[1], :]
        x = x + e 

        # forward pass
        x = self.transformer(x)

        # extract CLS token
        x = x[:, 0]

        # projection into embedding space followed by norm
        x = self.projection(x)
        return F.normalize(x, p=2, dim=-1)
