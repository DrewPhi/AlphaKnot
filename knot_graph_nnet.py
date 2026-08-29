#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.utils import to_dense_batch
import knot_graph_game as KnotGraphGame
import config
from pd_code_utils import PD_CANONICALIZATION_VERSION, deserialize_graph

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
        self.embed_crossing_state = nn.Embedding(3, hidden_dim)
        # A PD crossing is an ordered tuple (a, b, c, d).  Keep the four
        # position-specific representations separate until after concatenation:
        # summing E(label) + P(position) over the tuple makes the positional
        # contribution constant and therefore erases the PD ordering.
        self.crossing_projection = nn.Linear(4 * embed_dim, hidden_dim)

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

    def encode_crossings(self, node_ids):
        """Encode ordered PD tuples without treating their entries as a set."""
        N = node_ids.size(0)
        device = node_ids.device

        pos_idx = torch.arange(4, device=device).expand(N, 4)
        strand_embeds = self.embed_strand(node_ids)
        pos_embeds = self.embed_pos(pos_idx)
        ordered_slots = strand_embeds + pos_embeds
        return self.crossing_projection(ordered_slots.reshape(N, -1))

    def forward(self, data):
        batch = data.batch if hasattr(data, 'batch') else None

        node_ids = data.x[:, :4].long()
        crossing_state = data.x[:, 4].long()
        node_feat = (
            self.encode_crossings(node_ids)
            + self.embed_crossing_state(crossing_state)
        )

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


class CrossingStateMLP(nn.Module):
    """Fixed-shadow control model over ordered categorical crossing states.

    This intentionally ignores graph connectivity.  It tests whether the exact
    targets and optimization pipeline can fit the complete seven-crossing game
    when every dynamic state variable is directly accessible.
    """

    def __init__(self, game, hidden_dim=256, state_embed_dim=16):
        super().__init__()
        self.action_size = game.getActionSize()
        self.num_nodes = len(game.initial_pd_code)
        self.embed_crossing_state = nn.Embedding(3, state_embed_dim)
        input_dim = self.num_nodes * state_embed_dim
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, self.action_size)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        crossing_states = data.x[:, 4].long()
        if crossing_states.numel() % self.num_nodes != 0:
            raise ValueError(
                "Crossing-state MLP requires a fixed number of nodes per graph"
            )
        crossing_states = crossing_states.reshape(-1, self.num_nodes)
        features = self.embed_crossing_state(crossing_states).flatten(1)
        hidden = self.trunk(features)
        return self.policy_head(hidden), torch.tanh(self.value_head(hidden))


class PDStateMLP(nn.Module):
    """Dense shared-shadow control over ordered PD labels and game state.

    Unlike :class:`CrossingStateMLP`, this model can distinguish different
    source diagrams.  It deliberately uses the canonical numeric PD labels, so
    it is a capacity control rather than a relabeling-invariant architecture.
    """

    def __init__(
        self,
        game,
        hidden_dim=512,
        num_layers=4,
        label_embed_dim=32,
        state_embed_dim=32,
        dropout=0.0,
    ):
        super().__init__()
        self.action_size = game.getActionSize()
        self.num_nodes = len(game.initial_pd_code)
        self.embed_strand = nn.Embedding(config.max_strand_label + 1, label_embed_dim)
        self.embed_crossing_state = nn.Embedding(3, state_embed_dim)
        crossing_dim = 4 * label_embed_dim + state_embed_dim
        input_dim = self.num_nodes * crossing_dim
        layers = []
        for layer_index in range(num_layers):
            layers.append(nn.Linear(input_dim if layer_index == 0 else hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(hidden_dim, self.action_size)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        if data.x.size(0) % self.num_nodes != 0:
            raise ValueError("PD-state MLP requires a fixed crossing count")
        graphs = data.x.reshape(-1, self.num_nodes, data.x.size(1))
        pd_labels = graphs[:, :, :4].long()
        crossing_states = graphs[:, :, 4].long()
        label_features = self.embed_strand(pd_labels).flatten(2)
        state_features = self.embed_crossing_state(crossing_states)
        features = torch.cat([label_features, state_features], dim=2).flatten(1)
        hidden = self.trunk(features)
        return self.policy_head(hidden), torch.tanh(self.value_head(hidden))


class PortRelationLayer(nn.Module):
    """Typed message passing on the four half-edges of every PD crossing."""

    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.cyclic_next_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.cyclic_prev_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.opposite_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.arc_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = dropout

    def forward(self, ports, cyclic_next, cyclic_prev, opposite, arc_peer):
        messages = (
            self.self_linear(ports)
            + self.cyclic_next_linear(ports[cyclic_next])
            + self.cyclic_prev_linear(ports[cyclic_prev])
            + self.opposite_linear(ports[opposite])
            + self.arc_linear(ports[arc_peer])
        )
        messages = self.output_linear(F.gelu(messages))
        messages = F.dropout(messages, p=self.dropout, training=self.training)
        return self.norm(ports + messages)


class PortGraphTransformerNet(nn.Module):
    """PD-native local half-edge encoder plus global crossing attention."""

    def __init__(
        self,
        game,
        hidden_dim=128,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
        num_port_layers=4,
        position_mode="none",
        direct_state_residual=False,
        variable_size=False,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.action_size = game.getActionSize()
        self.num_nodes = len(game.initial_pd_code)
        self.hidden_dim = hidden_dim
        if position_mode not in {"none", "list", "pd-traversal", "pd-structural"}:
            raise ValueError(f"Unknown position mode: {position_mode}")
        self.position_mode = position_mode
        self.direct_state_residual = direct_state_residual
        self.variable_size = variable_size
        self.embed_slot = nn.Embedding(4, hidden_dim)
        self.embed_crossing_state = nn.Embedding(3, hidden_dim)
        self.embed_player = nn.Embedding(2, hidden_dim)
        self.embed_crossing_position = (
            nn.Embedding(self.num_nodes, hidden_dim)
            if position_mode in {"list", "pd-traversal"}
            else None
        )
        # The variable-size model cannot use a learned table indexed by a
        # fixed crossing count.  These four deterministic structural features
        # describe a crossing's canonical traversal location and graph size;
        # the projection is shared for every crossing count.
        self.structural_position_projection = (
            nn.Sequential(
                nn.Linear(4, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            if position_mode == "pd-structural"
            else None
        )
        self.port_layers = nn.ModuleList(
            PortRelationLayer(hidden_dim, dropout)
            for _ in range(num_port_layers)
        )
        self.crossing_projection = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.global_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        self.graph_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.graph_token, std=0.02)
        self.policy_head = nn.Linear(hidden_dim, 2)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _relation_indices(pd_labels, crossing_batch):
        """Return port-neighbor indices; PD integers only establish arc pairs."""
        device = pd_labels.device
        num_crossings = pd_labels.size(0)
        port_ids = torch.arange(4 * num_crossings, device=device).reshape(-1, 4)
        cyclic_next = port_ids[:, [1, 2, 3, 0]].reshape(-1)
        cyclic_prev = port_ids[:, [3, 0, 1, 2]].reshape(-1)
        opposite = port_ids[:, [2, 3, 0, 1]].reshape(-1)
        labels = pd_labels.reshape(-1)
        port_batch = crossing_batch.repeat_interleave(4)
        label_stride = labels.max() + 1
        grouping_key = port_batch * label_stride + labels
        sorted_keys, order = torch.sort(grouping_key)
        if sorted_keys.numel() % 2 != 0 or not bool(
            sorted_keys[0::2].eq(sorted_keys[1::2]).all()
        ):
            raise ValueError(
                "Every PD arc label must occur exactly twice within a graph"
            )
        arc_peer = torch.empty_like(order)
        arc_peer[order[0::2]] = order[1::2]
        arc_peer[order[1::2]] = order[0::2]
        return cyclic_next, cyclic_prev, opposite, arc_peer

    def _crossing_positions(self, pd_labels):
        """Return configured list positions or first-encounter PD traversal ranks."""
        if pd_labels.size(0) % self.num_nodes != 0:
            raise ValueError(
                "Positional encoding requires the configured crossing count"
            )
        num_graphs = pd_labels.size(0) // self.num_nodes
        if self.position_mode == "list":
            return torch.arange(
                self.num_nodes, device=pd_labels.device
            ).repeat(num_graphs)
        if self.position_mode == "pd-traversal":
            first_encounter = pd_labels.min(dim=1).values.reshape(
                num_graphs, self.num_nodes
            )
            traversal_order = first_encounter.argsort(dim=1)
            ranks = torch.empty_like(traversal_order)
            rank_values = torch.arange(
                self.num_nodes, device=pd_labels.device
            ).expand_as(traversal_order)
            ranks.scatter_(1, traversal_order, rank_values)
            return ranks.reshape(-1)
        raise RuntimeError("Crossing positions requested with position_mode='none'")

    @staticmethod
    def _structural_position_features(pd_labels, crossing_batch):
        """Return scale-free canonical PD traversal features per crossing.

        Canonical arc labels determine first-encounter rank within each graph.
        No learned lookup is indexed by the crossing count, so the encoding is
        defined for diagrams larger than those seen during training.
        """
        features = torch.empty(
            (pd_labels.size(0), 4),
            dtype=torch.float32,
            device=pd_labels.device,
        )
        first_encounter = pd_labels.min(dim=1).values
        for graph_index in range(int(crossing_batch.max().item()) + 1):
            node_indices = (crossing_batch == graph_index).nonzero(as_tuple=False).view(-1)
            count = node_indices.numel()
            order = first_encounter[node_indices].argsort()
            ranks = torch.empty(count, dtype=torch.long, device=pd_labels.device)
            ranks[order] = torch.arange(count, device=pd_labels.device)
            phase = ranks.float() / float(max(count, 1))
            linear = ranks.float() / float(max(count - 1, 1))
            size = torch.ones_like(linear) * torch.log1p(
                torch.tensor(float(count), device=pd_labels.device)
            )
            features[node_indices] = torch.stack(
                (
                    torch.sin(2 * torch.pi * phase),
                    torch.cos(2 * torch.pi * phase),
                    linear,
                    size,
                ),
                dim=1,
            )
        return features

    def forward(self, data):
        crossing_batch = (
            data.batch
            if hasattr(data, "batch") and data.batch is not None
            else torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
        )
        pd_labels = data.x[:, :4].long()
        crossing_state = data.x[:, 4].long()
        relations = self._relation_indices(pd_labels, crossing_batch)
        state_signal = self.embed_crossing_state(crossing_state)
        position_signal = None
        if self.embed_crossing_position is not None:
            position_signal = self.embed_crossing_position(
                self._crossing_positions(pd_labels)
            )
        elif self.structural_position_projection is not None:
            position_signal = self.structural_position_projection(
                self._structural_position_features(pd_labels, crossing_batch)
            )

        slots = torch.arange(4, device=data.x.device).expand(data.x.size(0), 4)
        port_features = self.embed_slot(slots) + state_signal.unsqueeze(1)
        if position_signal is not None:
            port_features = port_features + position_signal.unsqueeze(1)
        ports = port_features.reshape(-1, self.hidden_dim)
        for layer in self.port_layers:
            ports = layer(ports, *relations)

        crossings = self.crossing_projection(
            ports.reshape(data.x.size(0), 4 * self.hidden_dim)
        )
        if self.direct_state_residual:
            crossings = crossings + state_signal
        if position_signal is not None:
            crossings = crossings + position_signal
        dense_crossings, valid_crossings = to_dense_batch(
            crossings, crossing_batch
        )
        dense_states, _ = to_dense_batch(crossing_state, crossing_batch)
        resolved_count = ((dense_states != 0) & valid_crossings).sum(dim=1)
        player_index = resolved_count.remainder(2)
        graph_token = self.graph_token.expand(dense_crossings.size(0), -1, -1)
        graph_token = graph_token + self.embed_player(player_index).unsqueeze(1)
        sequence = torch.cat([graph_token, dense_crossings], dim=1)
        padding_mask = torch.cat(
            [
                torch.zeros(
                    (valid_crossings.size(0), 1),
                    dtype=torch.bool,
                    device=valid_crossings.device,
                ),
                ~valid_crossings,
            ],
            dim=1,
        )
        encoded = self.global_transformer(
            sequence, src_key_padding_mask=padding_mask
        )

        policy_by_crossing = self.policy_head(encoded[:, 1:])
        if not self.variable_size and policy_by_crossing.size(1) != self.num_nodes:
            raise ValueError(
                "Current AlphaZero wrapper requires the configured crossing count"
            )
        if self.variable_size:
            policy = policy_by_crossing.reshape(policy_by_crossing.size(0), -1)
        else:
            policy = policy_by_crossing.reshape(-1, self.action_size)
        value = torch.tanh(self.value_head(encoded[:, 0]))
        return policy, value


class NNetWrapper:
    def __init__(
        self,
        game,
        hidden_dim=64,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        device=None,
        architecture="graph",
    ):
        """
        Wrapper for the KnotGraphNet to interface with AlphaZero-General training.
        """
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.architecture = architecture
        if architecture == "graph":
            model = KnotGraphNet(game, hidden_dim, num_heads, num_layers, dropout)
        elif architecture == "crossing-mlp":
            model = CrossingStateMLP(game, hidden_dim=hidden_dim)
        elif architecture == "pd-state-mlp":
            model = PDStateMLP(
                game,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif architecture == "port-graph-transformer":
            model = PortGraphTransformerNet(
                game,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif architecture in {
            "port-transformer-residual",
            "port-transformer-indexed",
            "port-transformer-pd-position",
            "variable-port-transformer",
        }:
            position_mode = {
                "port-transformer-residual": "none",
                "port-transformer-indexed": "list",
                "port-transformer-pd-position": "pd-traversal",
                "variable-port-transformer": "pd-structural",
            }[architecture]
            model = PortGraphTransformerNet(
                game,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                position_mode=position_mode,
                direct_state_residual=True,
                variable_size=architecture == "variable-port-transformer",
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        self.model = model.to(self.device)
        self.action_size = game.getActionSize()
        self.optimizer = optim.Adam(self.model.parameters(), lr=getattr(game, 'learning_rate', 0.001))
        print("[Device]", self.device)

        # Optional: resume training or load existing model
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

        # Optional: precompute PD structure mappings
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
                elif isinstance(state, dict):
                    data = deserialize_graph(state)
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

                total_loss += loss.item()
                count += 1
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

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
            'latest_loss': getattr(self, 'latest_loss', float('inf')),
            'architecture': self.architecture,
            'pd_canonicalization_version': PD_CANONICALIZATION_VERSION,
        }, filepath)
        print(f"Checkpoint saved at {filepath}")

    def load_checkpoint(self, filename, load_optimizer=True):
        assert os.path.isfile(filename), f"No model found at {filename}"
        checkpoint = torch.load(filename, map_location=self.device)

        saved_architecture = checkpoint.get('architecture', 'graph')
        if saved_architecture != self.architecture:
            raise ValueError(
                f"Checkpoint architecture is {saved_architecture!r}, but "
                f"wrapper architecture is {self.architecture!r}"
            )
        saved_canonicalization = checkpoint.get(
            'pd_canonicalization_version', 'legacy-uncanonicalized'
        )
        if saved_canonicalization != PD_CANONICALIZATION_VERSION:
            raise ValueError(
                "Checkpoint PD canonicalization is "
                f"{saved_canonicalization!r}, but runtime requires "
                f"{PD_CANONICALIZATION_VERSION!r}"
            )

        # Get the state dict
        state_dict = checkpoint['state_dict']

        # If model is wrapped in DistributedDataParallel
        is_ddp = isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
        new_state_dict = {}

        for k, v in state_dict.items():
            if is_ddp and not k.startswith("module."):
                new_state_dict["module." + k] = v
            elif not is_ddp and k.startswith("module."):
                new_state_dict[k[len("module."):]] = v
            else:
                new_state_dict[k] = v

        self.model.load_state_dict(new_state_dict)
        if load_optimizer and 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.latest_loss = checkpoint.get('latest_loss', float('inf'))
        print(f"[Device] Loaded checkpoint from {filename}")
