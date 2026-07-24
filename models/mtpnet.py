import torch
from torch import nn


class MechanismFeatureEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.feature_gate = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.projection(x)
        gate = self.feature_gate(x)
        return encoded * gate


class TemporalResidualBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.mix = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.mix(self.norm(x)))


class MTPNetRiskPredictor(nn.Module):
    """Mechanism Temporal Prediction Net for regional future-risk prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        head_hidden_size = max(1, hidden_size // 2)
        self.mechanism_encoder = MechanismFeatureEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
        )
        self.temporal_encoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.temporal_refine = TemporalResidualBlock(
            hidden_size=hidden_size,
            dropout=dropout,
        )
        self.temporal_attention = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, head_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.mechanism_encoder(x)
        temporal_output, _ = self.temporal_encoder(encoded)
        temporal_output = self.temporal_refine(temporal_output)
        attention = torch.softmax(self.temporal_attention(temporal_output), dim=1)
        pooled = torch.sum(temporal_output * attention, dim=1)
        return self.head(pooled)
