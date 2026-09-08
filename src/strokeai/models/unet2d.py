from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet2D(nn.Module):
    """Small 2-D U-Net (≈1.9 M params at base=16, depth=4). Input (B, in_ch, H, W) -> logits (B, 1, H, W)."""

    def __init__(self, in_ch: int = 2, base: int = 16, depth: int = 4, out_ch: int = 1):
        super().__init__()
        chs = [base * 2**i for i in range(depth + 1)]
        self.enc = nn.ModuleList([conv_block(in_ch if i == 0 else chs[i - 1], chs[i]) for i in range(depth)])
        self.bottleneck = conv_block(chs[depth - 1], chs[depth])
        self.up = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([conv_block(chs[i] * 2, chs[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(chs[0], out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc in self.enc:
            x = enc(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([x, skip], dim=1))
        return self.head(x)
