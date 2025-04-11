#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv, global_mean_pool
import knot_graph_game as KnotGraphGame
import config

class KnotGraphNet(nn.Module):
    def __init__(self, game, hidden_dim=64, num_heads=8, num_layers=6, dropout=0.1):
        """
        Graph Transformer model for the knot resolution game.
        :param game: The KnotGame instance (provides initial PD code and action size).
        :param hidden_dim: Dimension of node feature embeddings and transformer hidden size.
        :param num_heads: Number of attention heads in TransformerConv.
        :param num_layers: Number of TransformerConv layers.
        :param dropout: Dropout rate for attention layers.
        """
        super().__init__()
        self.action_size = game.getActionSize()
        self.num_nodes = len(game.initial_pd_code)
        self.num_layers = num_layers
        self.dropout = dropout

        # Precompute mapping from strand labels for embedding
        label_to_nodes = {}
        for i, crossing in enumerate(game.initial_pd_code):
            for label in crossing[:4]:
                label = int(label)
                label_to_nodes.setdefault(label, []).append(i)

        max_label = config.max_strand_label
        embed_dim = hidden_dim

        # Node/strand embeddings
        self.embed_strand = nn.Embedding(max_label + 1, embed_dim)
        self.embed_pos = nn.Embedding(4, embed_dim)

        # Edge feature embedding
        edge_sign_dim = 8
        self.embed_edge_sign = nn.Embedding(3, edge_sign_dim)

        edge_in_dim = embed_dim + edge_sign_dim
        edge_out_dim = 16
        self.edge_linear = nn.Linear(edge_in_dim, edge_out_dim)

        # TransformerConv layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer in range(num_layers):
            out_channels = hidden_dim // num_heads if num_heads > 1 else hidden_dim
            concat = True
            conv = TransformerConv(
                in_channels=hidden_dim,
                out_channels=out_channels,
                heads=num_heads,
                concat=concat,
                dropout=self.dropout,
                edge_dim=edge_out_dim,
                bias=True
            )
            self.convs.append(conv)
            if layer < num_layers - 1:
                self.norms.append(nn.LayerNorm(hidden_dim))

        # Output heads
        self.policy_head = nn.Linear(hidden_dim, 2)
        self.value_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.value_fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        batch = data.batch if hasattr(data, 'batch') else None

        node_ids = data.x[:, :4].long()
        N = node_ids.size(0)
        device = node_ids.device

        pos_idx = torch.arange(4, device=device).expand(N, 4)
        strand_embeds = self.embed_strand(node_ids)
        pos_embeds = self.embed_pos(pos_idx)
        strand_embeds = strand_embeds + pos_embeds
        node_feat = strand_embeds.sum(dim=1)

        x = node_feat

        edge_index = data.edge_index
        edge_attr = data.edge_attr
        edge_label = edge_attr[:, 0].long()
        edge_sign = edge_attr[:, 1].long()

        strand_e = self.embed_strand(edge_label)
        sign_e = self.embed_edge_sign(edge_sign)
        edge_feats = torch.cat([strand_e, sign_e], dim=-1)
        edge_feats = self.edge_linear(edge_feats)

        for li, conv in enumerate(self.convs):
            x_updated = conv(x, edge_index, edge_feats)
            if li < self.num_layers - 1:
                x = self.norms[li](x_updated)
                x = F.relu(x)
                if self.dropout > 1e-6:
                    x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = x_updated

        # Policy head
        node_policy = self.policy_head(x)
        if batch is None or batch.max() == 0:
            policy = node_policy.view(1, -1)
        else:
            num_graphs = batch.max().item() + 1
            policy_list = []
            for g in range(num_graphs):
                mask = (batch == g)
                nodes_g = node_policy[mask]
                policy_list.append(nodes_g.view(-1))
            policy = torch.stack(policy_list, dim=0)

        # Value head
        if batch is not None and batch.max().item() + 1 > 1:
            graph_feat = global_mean_pool(x, batch)
        else:
            graph_feat = x.mean(dim=0, keepdim=True)

        v = F.relu(self.value_fc1(graph_feat))
        v = torch.tanh(self.value_fc2(v))

        return policy, v


class NNetWrapper:
    def __init__(self, game, hidden_dim=64, num_heads=8, num_layers=6, dropout=0.1):
        """
        Wrapper for the KnotGraphNet to interface with AlphaZero-General training.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = KnotGraphNet(game, hidden_dim, num_heads, num_layers, dropout).to(self.device)
        self.action_size = game.getActionSize()
        self.optimizer = optim.Adam(self.model.parameters(), lr=getattr(game, 'learning_rate', 0.001))
        print("[Device]", self.device)

        # Load checkpoint if requested
        if hasattr(game, "resumeTraining") and game.resumeTraining:
            checkpoint_path = os.path.join(config.checkpoint, 'best.pth.tar')
            if os.path.isfile(checkpoint_path):
                print(f"Resuming training from {checkpoint_path}")
                self.load_checkpoint(checkpoint_path)
        elif hasattr(game, "load_model") and game.load_model:
            load_path = os.path.join(game.checkpoint_path, game.load_model_file)
            if os.path.isfile(load_path):
                print(f"Loading model from {load_path}")
                self.load_checkpoint(load_path)

        # Precompute mapping
        self.initial_pd_code = game.initial_pd_code
        self.num_nodes = len(game.initial_pd_code)
        self.label_to_nodes = {}
        for i, crossing in enumerate(game.initial_pd_code):
            for label in crossing[:4]:
                label = int(label)
                self.label_to_nodes.setdefault(label, []).append(i)
        self.initial_node_labels = [list(map(int, cr[:4])) for cr in game.initial_pd_code]

    def train(self, examples):
        """
        Train the network for a number of epochs on the provided examples.
        :param examples: list of (state, pi, v) tuples.
        """
        self.latest_loss = 0
        self.model.train()
        epochs = getattr(config, 'num_epochs', 1)

        for epoch in range(epochs):
            total_loss = 0
            count = 0
            for state, pi, v in examples:
                if isinstance(state, Data):
                    data = state
                else:
                    data = KnotGraphGame.KnotGraphGame().pd_code_to_graph_data(state)

                data = data.to(self.device)
                target_pi = torch.tensor(pi, dtype=torch.float32).unsqueeze(0).to(self.device)
                target_v = torch.tensor([v], dtype=torch.float32).to(self.device)

                out_pi, out_v = self.model(data)
                log_probs = F.log_softmax(out_pi, dim=1)
                l_pi = -torch.sum(target_pi * log_probs)
                l_v = F.mse_loss(out_v.view(-1), target_v.view(-1))
                loss = l_pi + l_v

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                count += 1

            self.latest_loss = total_loss / count if count > 0 else float('inf')

    def predict(self, state):
        self.model.eval()
        if isinstance(state, Data):
            data = state
        else:
            data = KnotGraphGame.KnotGraphGame().pd_code_to_graph_data(state)

        data = data.to(self.device)
        with torch.no_grad():
            out_pi, out_v = self.model(data)
            pi_probs = F.softmax(out_pi, dim=1).cpu().numpy()[0]
            v = out_v.item()
        return pi_probs, v

    def save_checkpoint(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'latest_loss': getattr(self, 'latest_loss', float('inf'))
        }, filepath)
        print(f"Checkpoint saved at {filepath}")

    def load_checkpoint(self, filepath):
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"No checkpoint found at {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.latest_loss = checkpoint.get('latest_loss', float('inf'))
        self.model.to(self.device)
        print(f"Loaded checkpoint from {filepath}")
