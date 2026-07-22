import torch
from torch import nn


class RecurrentRiskPredictor(nn.Module):
    recurrent_class: type[nn.Module]

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = self.recurrent_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.encoder(x)
        return self.head(output[:, -1, :])


class RNNRiskPredictor(RecurrentRiskPredictor):
    recurrent_class = nn.RNN


class GRURiskPredictor(RecurrentRiskPredictor):
    recurrent_class = nn.GRU


class LSTMRiskPredictor(RecurrentRiskPredictor):
    recurrent_class = nn.LSTM
