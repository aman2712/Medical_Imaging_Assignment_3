"""Bond-aware message-passing network for graph-level binary classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool


class AtomEncoder(nn.Module):
    def __init__(self, category_sizes, hidden_channels):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(size, hidden_channels) for size in category_sizes]
        )
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight)

    def forward(self, x):
        out = 0
        for i, embedding in enumerate(self.embeddings):
            out = out + embedding(x[:, i])
        return out


class BondEncoder(nn.Module):
    def __init__(self, category_sizes, hidden_channels):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(size, hidden_channels) for size in category_sizes]
        )
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight)

    def forward(self, edge_attr):
        out = 0
        for i, embedding in enumerate(self.embeddings):
            out = out + embedding(edge_attr[:, i])
        return out


class GNN(nn.Module):
    def __init__(self, atom_category_sizes, bond_category_sizes, hidden_channels=64,
                 num_layers=3, dropout=0.2):
        super().__init__()
        self.dropout = dropout

        self.atom_encoder = AtomEncoder(atom_category_sizes, hidden_channels)
        self.bond_encoders = nn.ModuleList(
            [BondEncoder(bond_category_sizes, hidden_channels) for _ in range(num_layers)]
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, 2 * hidden_channels),
                nn.ReLU(),
                nn.Linear(2 * hidden_channels, hidden_channels),
            )
            self.convs.append(GINEConv(mlp, train_eps=True))
            self.norms.append(nn.BatchNorm1d(hidden_channels))

        self.head = nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x = self.atom_encoder(data.x)

        for conv, norm, bond_encoder in zip(self.convs, self.norms, self.bond_encoders):
            edge_embedding = bond_encoder(data.edge_attr)
            x = conv(x, data.edge_index, edge_embedding)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        pooled = global_mean_pool(x, data.batch)
        return self.head(pooled).view(-1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
