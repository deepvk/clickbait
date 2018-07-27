import torch.nn as nn


class PositionWise(nn.Module):
    def __init__(self, size, inner_size, dropout=0.1):
        super(PositionWise, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(size, inner_size),
            nn.ReLU(),
            nn.Linear(inner_size, size)
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(size, eps=1e-12)

    def forward(self, input):
        residual = input

        result = self.fc(input)
        result = self.dropout(result)

        return self.layer_norm(result + residual)
