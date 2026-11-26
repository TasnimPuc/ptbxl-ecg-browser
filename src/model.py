"""Compact residual 1D-CNN with depthwise-separable convolutions (~0.1 M parameters, < 1 MB as ONNX).
Per-lead normalisation is a buffer INSIDE the model, so the browser feeds raw millivolts: input [batch, 12, 1000]."""
from __future__ import annotations

import torch
import torch.nn as nn


class SepBlock(nn.Module):
    def __init__(self, cin, cout, k, stride, dropout):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(cin, cin, k, stride=stride, padding=k // 2, groups=cin, bias=False), nn.BatchNorm1d(cin),
            nn.ReLU(), nn.Conv1d(cin, cout, 1, bias=False), nn.BatchNorm1d(cout), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(cout, cout, k, padding=k // 2, groups=cout, bias=False), nn.BatchNorm1d(cout),
            nn.Conv1d(cout, cout, 1, bias=False), nn.BatchNorm1d(cout))
        self.skip = nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm1d(cout)) \
            if (cin != cout or stride != 1) else nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class ECGNet(nn.Module):
    def __init__(self, n_leads=12, n_classes=5, widths=(32, 64, 96, 128), k=7, dropout=0.2):
        super().__init__()
        self.register_buffer("mu", torch.zeros(1, n_leads, 1))
        self.register_buffer("sd", torch.ones(1, n_leads, 1))
        self.stem = nn.Sequential(nn.Conv1d(n_leads, widths[0], 15, stride=2, padding=7, bias=False),
                                  nn.BatchNorm1d(widths[0]), nn.ReLU())
        blocks, c = [], widths[0]
        for w in widths:
            blocks.append(SepBlock(c, w, k, 2, dropout))
            c = w
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * c, n_classes))

    def forward(self, x):                        # x: raw mV [B, 12, T]
        x = (x - self.mu) / self.sd
        h = self.blocks(self.stem(x))
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], 1))


class Probabilities(nn.Module):
    """Export wrapper: sigmoid probabilities instead of logits."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return torch.sigmoid(self.m(x))


def n_params(m) -> int:
    return sum(p.numel() for p in m.parameters())
