import torch
import torch.nn as nn

class TabularEvasionMLP(nn.Module):
    def __init__(self, num_numerical, categorical_cardinalities, emb_dim=8, hidden_dim=64):
        super(TabularEvasionMLP, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, emb_dim) for cardinality in categorical_cardinalities
        ])
        total_input_dim = num_numerical + (len(categorical_cardinalities) * emb_dim)
        self.net = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward_fused(self, x_fused):
        return self.net(x_fused)