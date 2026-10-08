"""Fully connected, scene-isolated GATv2, retaining self edges."""
import torch
from torch import nn
from torch.nn import functional as F


class SceneGATv2(nn.Module):
    def __init__(self, width=64, heads=4):
        super().__init__()
        self.heads = heads
        self.dim = width // heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.edge = nn.Linear(3, width, bias=False)
        self.attention = nn.Parameter(torch.empty(heads, self.dim))
        nn.init.xavier_uniform_(self.attention)
        self.out = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, hidden, positions, boundaries):
        bounds = boundaries.tolist()
        counts = [end - start for start, end in bounds]
        width = max(counts)
        indices = torch.tensor(
            [start + i if i < count else 0
             for (start, _), count in zip(bounds, counts) for i in range(width)],
            device=hidden.device,
        ).reshape(len(bounds), width)
        valid = torch.arange(width, device=hidden.device)[None] < torch.tensor(
            counts, device=hidden.device
        )[:, None]
        steps, agents, channels = hidden.shape
        x = hidden[:, indices]
        pos = positions[:, indices]
        shape = (steps, len(bounds), width, self.heads, self.dim)
        q, k, v = (layer(x).reshape(shape) for layer in (self.query, self.key, self.value))
        edge = self.edge(pos[:, :, :, None] - pos[:, :, None, :]).reshape(
            steps, len(bounds), width, width, self.heads, self.dim
        )
        pair = F.leaky_relu(q[:, :, :, None] + k[:, :, None, :] + edge, 0.2)
        scores = (pair * self.attention).sum(-1)
        scores = scores.masked_fill(~valid[None, :, None, :, None], -torch.inf)
        weights = scores.softmax(dim=3)
        pooled = torch.einsum("tsijh,tsjhd->tsihd", weights, v).reshape(
            steps, len(bounds), width, channels
        )
        return self.norm(hidden + self.out(pooled[:, valid].reshape(steps, agents, channels)))
