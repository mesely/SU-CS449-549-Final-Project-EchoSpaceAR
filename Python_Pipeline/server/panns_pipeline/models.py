"""Minimal PANNs-style CNN14 semantic model."""

from __future__ import annotations

try:
    import torch
    from torch import nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class ConvBlock(nn.Module):
        """Two-layer convolutional block used by the CNN14 family."""

        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.relu_(self.bn1(self.conv1(x)))
            x = F.relu_(self.bn2(self.conv2(x)))
            return F.avg_pool2d(x, kernel_size=2)


    class PannsCNN14Semantic(nn.Module):
        """CNN14-style semantic head for the EchoSpace label space."""

        def __init__(self, num_classes: int, dropout: float = 0.2, embedding_dim: int = 2048) -> None:
            super().__init__()
            self.bn0 = nn.BatchNorm2d(1)
            self.blocks = nn.ModuleList(
                [
                    ConvBlock(1, 64),
                    ConvBlock(64, 128),
                    ConvBlock(128, 256),
                    ConvBlock(256, 512),
                    ConvBlock(512, 1024),
                    ConvBlock(1024, 2048),
                ]
            )
            self.dropout = float(dropout)
            self.fc1 = nn.Linear(2048, embedding_dim)
            self.classifier = nn.Linear(embedding_dim, num_classes)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            if x.ndim != 3:
                raise ValueError(f"Expected [batch, frames, mel], got rank={x.ndim}")
            x = x.unsqueeze(1)
            x = self.bn0(x)
            for block in self.blocks:
                x = block(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

            x = torch.mean(x, dim=3)
            x_mean = torch.mean(x, dim=2)
            x_max = torch.max(x, dim=2).values
            embedding = F.relu_(self.fc1(x_mean + x_max))
            embedding = F.dropout(embedding, p=self.dropout, training=self.training)
            logits = self.classifier(embedding)
            return logits, embedding


else:

    class PannsCNN14Semantic:  # pragma: no cover - optional dependency
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PyTorch is required to instantiate PannsCNN14Semantic.")
