# neural_network.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import config

# Residual 1D Block
class ResidualBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3, dropout=0.3):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class KnotNNet(nn.Module):
    def __init__(self, game):
        super(KnotNNet, self).__init__()
        self.board_length = len(game.get_canonical_form())  # 7 in your case
        self.input_channels = 1  # flat signal, not multichannel
        self.action_size = game.get_action_size()
        self.num_blocks = 3
        self.hidden_channels = 64

        self.input_conv = nn.Conv1d(self.input_channels, self.hidden_channels, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm1d(self.hidden_channels)
        self.relu = nn.ReLU()

        self.res_blocks = nn.Sequential(*[
            ResidualBlock1D(self.hidden_channels, kernel_size=3, dropout=0.3)
            for _ in range(self.num_blocks)
        ])

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # (B, C, 1)
        self.shared_fc = nn.Linear(self.hidden_channels, 512)

        self.policy_head = nn.Linear(512, self.action_size)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        x = x.view(-1, 1, self.board_length)  # (B, C=1, L)
        x = self.relu(self.input_bn(self.input_conv(x)))
        x = self.res_blocks(x)
        x = self.global_avg_pool(x).squeeze(-1)  # (B, C)
        x = self.relu(self.shared_fc(x))
        policy = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        return policy, value


class NNetWrapper:
    def __init__(self, game):
        self.nnet = KnotNNet(game)
        self.board_size = len(game.get_canonical_form())
        self.action_size = game.get_action_size()
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=config.learning_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nnet.to(self.device)

        # Resume / load model if indicated
        if config.resumeTraining:
            model_path = 'championModel/best.pth.tar'
            if os.path.isfile(model_path):
                print(f"Resuming training from {model_path}")
                self.load_checkpoint(model_path)
            else:
                print("WARNING: resumeTraining=True but no checkpoint found at championModel/best.pth.tar. Starting fresh.")
        elif config.load_model:
            load_path = os.path.join(config.checkpoint_path, config.load_model_file)
            print(f"Loading model from specified file: {load_path}")
            self.load_checkpoint(load_path)

    def train(self, examples):
        self.nnet.train()
        for epoch in range(config.num_epochs):
            for state, pi, v in examples:
                self.optimizer.zero_grad()

                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                target_pi = torch.tensor(pi, dtype=torch.float32).unsqueeze(0).to(self.device)
                target_v = torch.tensor([v], dtype=torch.float32).to(self.device)

                out_pi, out_v = self.nnet(state)
                l_pi = -torch.sum(target_pi * torch.log_softmax(out_pi, dim=1))
                l_v = nn.functional.mse_loss(out_v.view(-1), target_v)
                loss = l_pi + l_v

                loss.backward()
                self.optimizer.step()

    def predict(self, state):
        self.nnet.eval()
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            pi, v = self.nnet(state)
            pi = nn.functional.softmax(pi, dim=1).cpu().numpy()[0]
            v = v.item()
        return pi, v

    def save_checkpoint(self, filename):
        torch.save({
            'state_dict': self.nnet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, filename)

    def load_checkpoint(self, filename):
        if os.path.isfile(filename):
            print(f"Loading model from {filename}")
            checkpoint = torch.load(filename, map_location=self.device)
            self.nnet.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Model loaded successfully.")
        else:
            print(f"No model found at {filename}, starting from scratch.")
