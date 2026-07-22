import torch
from torch import nn


class SelectiveSSMBlock(nn.Module):
    """A small Mamba-inspired selective state-space block implemented in pure PyTorch."""

    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.input_projection = nn.Linear(hidden_size, hidden_size * 3)
        self.depthwise_conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=3,
            padding=1,
            groups=hidden_size,
        )
        self.state_decay = nn.Parameter(torch.zeros(hidden_size))
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        candidate, gate, delta = self.input_projection(x).chunk(3, dim=-1)
        candidate = self.depthwise_conv(candidate.transpose(1, 2)).transpose(1, 2)
        candidate = torch.tanh(candidate)
        gate = torch.sigmoid(gate)
        delta = torch.sigmoid(delta)
        base_decay = torch.sigmoid(self.state_decay).view(1, -1)

        state = torch.zeros(x.size(0), x.size(2), dtype=x.dtype, device=x.device)
        outputs = []
        for step in range(x.size(1)):
            decay = base_decay * delta[:, step, :]
            state = decay * state + (1.0 - decay) * candidate[:, step, :]
            outputs.append((gate[:, step, :] * state).unsqueeze(1))

        y = torch.cat(outputs, dim=1)
        y = self.output_projection(y)
        return residual + self.dropout(y)


class MambaRiskPredictor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.blocks = nn.ModuleList(
            [SelectiveSSMBlock(hidden_size, dropout=dropout) for _ in range(num_layers)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x[:, -1, :])
