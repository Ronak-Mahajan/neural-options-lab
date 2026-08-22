"""Residual MLP surrogate pricer.

Architecture notes
------------------
- Inputs (m, T, sigma, r) are affinely mapped to [-1, 1] using the training
  box, which keeps every feature on the same scale without a fitted scaler.
- SiLU activations + LayerNorm residual blocks: smooth (C-infinity)
  activations matter here because we differentiate the network to obtain
  Greeks - ReLU would give piecewise-constant delta and zero gamma a.e.
- Softplus output enforces price positivity while staying smooth.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc2(self.act(self.fc1(self.norm(x))))
        return x + h


class AsianPricerNet(nn.Module):
    """Maps normalized (moneyness, maturity, sigma, rate) -> call price / K."""

    def __init__(self, in_features: int = 4, width: int = 128,
                 n_blocks: int = 4):
        super().__init__()
        self.stem = nn.Linear(in_features, width)
        self.blocks = nn.Sequential(*[ResidualBlock(width)
                                      for _ in range(n_blocks)])
        self.head = nn.Linear(width, 1)
        self.act = nn.SiLU()
        self.out = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.stem(x))
        h = self.blocks(h)
        return self.out(self.head(h)).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
