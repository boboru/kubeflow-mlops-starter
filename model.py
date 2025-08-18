import torch
import torch.nn as nn

class DCNv2(nn.Module):
    def __init__(self, dense_dim, sparse_cardinalities, embed_dim=8, cross_layers=2, dnn_layers=[128, 64]):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, embed_dim) for card in sparse_cardinalities
        ])
        self.embed_dim = embed_dim
        self.num_sparse = len(sparse_cardinalities)
        self.input_dim = dense_dim + embed_dim * self.num_sparse

        # CrossNetworkV2 inline
        self.cross_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_dim, self.input_dim),
                nn.LayerNorm(self.input_dim)
            ) for _ in range(cross_layers)
        ])

        # DNN layers
        dnn = []
        dnn_input_dim = self.input_dim
        for dim in dnn_layers:
            dnn.append(nn.Linear(dnn_input_dim, dim))
            dnn.append(nn.ReLU())
            dnn_input_dim = dim
        self.dnn = nn.Sequential(*dnn)

        # Output
        self.output = nn.Linear(self.input_dim + dnn_layers[-1], 1)

    def forward(self, dense, sparse):
        # embedding lookup
        emb = [emb_layer(sparse[:, i]) for i, emb_layer in enumerate(self.embeddings)]
        sparse_embed = torch.cat(emb, dim=1)
        x = torch.cat([dense, sparse_embed], dim=1)

        # cross network
        cross_out = x
        for layer in self.cross_layers:
            cross_out = layer(cross_out) + cross_out

        # DNN
        dnn_out = self.dnn(x)

        # concat and output
        combined = torch.cat([cross_out, dnn_out], dim=1)
        return torch.sigmoid(self.output(combined))
