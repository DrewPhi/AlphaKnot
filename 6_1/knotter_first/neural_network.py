import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import config
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3, dropout=0.3):
        super(ResidualBlock1D, self).__init__()
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=0)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=0)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        # Circular padding manually before each convolution
        pad_amt = self.kernel_size // 2
        x_padded = F.pad(x, (pad_amt, pad_amt), mode='circular')

        out = self.conv1(x_padded)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out_padded = F.pad(out, (pad_amt, pad_amt), mode='circular')
        out = self.conv2(out_padded)
        out = self.bn2(out)

        out += x  # Residual connection
        return self.relu(out)

class KnotNNet(nn.Module):
    def __init__(self, game):
        super(KnotNNet, self).__init__()
        self.num_crossings = len(game.initial_pd_code)
        self.input_features = 5  # each mirror image has 5 values
        self.hidden_channels = 64
        self.num_blocks = 3

        self.initial_conv = nn.Conv1d(self.input_features, self.hidden_channels, kernel_size=2, stride=2)
        self.initial_bn = nn.BatchNorm1d(self.hidden_channels)
        self.relu = nn.ReLU()

        self.res_blocks = nn.Sequential(*[
            ResidualBlock1D(self.hidden_channels, kernel_size=3, dropout=0.3)
            for _ in range(self.num_blocks)
        ])

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Shared fully connected for decision heads
        self.crossing_fc = nn.Linear(self.hidden_channels * 2, 128)
        self.crossing_out = nn.Linear(128, self.num_crossings)

        self.resolution_fc = nn.Linear(self.hidden_channels * 2, 128)
        self.resolution_out = nn.Linear(128, 2)

        self.value_head = nn.Linear(self.hidden_channels * 2, 1)

    def forward(self, x):
        # x: (B, num_crossings, 2, 5) → (B, 5, 2 * num_crossings)
        B, N, _, F = x.shape
        x = x.view(B, N * 2, F).permute(0, 2, 1)  # (B, F, 2N)

        x = self.relu(self.initial_bn(self.initial_conv(x)))  # (B, C, N)
        local_feat = self.res_blocks(x)  # (B, C, N)

        global_feat = self.global_pool(local_feat)  # (B, C, 1)
        global_feat = global_feat.expand(-1, -1, local_feat.size(2))  # (B, C, N)

        combined = torch.cat([local_feat, global_feat], dim=1)  # (B, 2C, N)
        pooled = self.global_pool(combined).squeeze(-1)  # (B, 2C)

        # Decision heads
        crossing_logits = self.crossing_out(self.relu(self.crossing_fc(pooled)))
        resolution_logits = self.resolution_out(self.relu(self.resolution_fc(pooled)))
        value = torch.tanh(self.value_head(pooled))
        return crossing_logits, resolution_logits, value

class NNetWrapper:
    def __init__(self, game):
        self.nnet = KnotNNet(game)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nnet.to(self.device)
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=config.learning_rate)
        self.last_training_loss = None
        self.num_crossings = len(game.initial_pd_code)

    def train(self, examples):
        self.nnet.train()
        total_loss = 0
        for epoch in range(config.num_epochs):
            for state, (pi_crossing, pi_res), v in examples:
                self.optimizer.zero_grad()
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                pi_c = torch.tensor(pi_crossing, dtype=torch.float32).unsqueeze(0).to(self.device)
                pi_r = torch.tensor(pi_res, dtype=torch.float32).unsqueeze(0).to(self.device)
                target_v = torch.tensor([v], dtype=torch.float32).to(self.device)

                out_c, out_r, out_v = self.nnet(state)
                l_c = -torch.sum(pi_c * torch.log_softmax(out_c, dim=1))
                l_r = -torch.sum(pi_r * torch.log_softmax(out_r, dim=1))
                l_v = nn.functional.mse_loss(out_v.view(-1), target_v)
                loss = l_c + l_r + l_v

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        self.last_training_loss = total_loss / max(1, len(examples))

    def predict(self, state):
        self.nnet.eval()
        with torch.no_grad():
          #  print("INPUT STATE TO NN (SHAPE):", np.array(state).shape)
          #  print("INPUT STATE TO NN (SLICE):", np.array(state)[:2])  # print first 2 crossings for brevity

            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            cross_logits, res_logits, value = self.nnet(state)
            cross_probs = nn.functional.softmax(cross_logits, dim=1).cpu().numpy()[0]
            res_probs = nn.functional.softmax(res_logits, dim=1).cpu().numpy()[0]
            return cross_probs, res_probs, value.item()


    def save_checkpoint(self, filename):
        torch.save({
            'state_dict': self.nnet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, filename)

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.nnet.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])