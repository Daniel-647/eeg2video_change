# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['XLMRoberta', 'xlm_roberta_large']


class SelfAttention(nn.Module):

    def __init__(self, dim, num_heads, dropout=0.1, eps=1e-5):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        x:   [B, L, C].
        """
        b, s, c, n, d = *x.size(), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.q(x).reshape(b, s, n, d).permute(0, 2, 1, 3)
        k = self.k(x).reshape(b, s, n, d).permute(0, 2, 1, 3)
        v = self.v(x).reshape(b, s, n, d).permute(0, 2, 1, 3)

        # compute attention
        p = self.dropout.p if self.training else 0.0
        x = F.scaled_dot_product_attention(q, k, v, mask, p)
        x = x.permute(0, 2, 1, 3).reshape(b, s, c)

        # output
        x = self.o(x)
        x = self.dropout(x)
        return x


class AttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.1, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.eps = eps

        # layers
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = SelfAttention(dim, num_heads, dropout, eps)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout))

    def forward(self, x, mask):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class XLMRoberta(nn.Module):

    def __init__(self, vocab_size, dim, num_heads, num_layers, mlp_ratio=4,
                 max_len=512, dropout=0.1, eps=1e-5, pad_id=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_ratio = mlp_ratio
        self.max_len = max_len
        self.pad_id = pad_id

        # embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = nn.Embedding(max_len, dim)
        self.dropout = nn.Dropout(dropout)

        # transformer
        self.blocks = nn.ModuleList([
            AttentionBlock(dim, num_heads, mlp_ratio, dropout, eps)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, ids):
        # embeddings
        b, s = ids.size()
        pos = torch.arange(s, device=ids.device).unsqueeze(0).expand(b, -1)
        x = self.token_embedding(ids) + self.pos_embedding(pos)
        x = self.dropout(x)

        # mask
        mask = ids.ne(self.pad_id).unsqueeze(1).unsqueeze(2)

        # transformer
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        return x


def xlm_roberta_large():
    return XLMRoberta(
        vocab_size=250002,
        dim=1024,
        num_heads=16,
        num_layers=24,
        mlp_ratio=4,
        max_len=514,
        dropout=0.1,
        eps=1e-5,
        pad_id=1)