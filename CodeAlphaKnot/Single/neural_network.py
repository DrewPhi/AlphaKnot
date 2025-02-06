# neural_network.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import config  

class KnotNNet(nn.Module):
    def __init__(self, game):
        super(KnotNNet, self).__init__()
        self.input_size = len(game.get_canonical_form())
        self.action_size = game.get_action_size()
        self.fc1 = nn.Linear(self.input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_policy = nn.Linear(64, self.action_size)
        self.fc_value = nn.Linear(64, 1)

    def forward(self, x):
        x = x.float()
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        policy = self.fc_policy(x)
        value = torch.tanh(self.fc_value(x))
        return policy, value

class NNetWrapper:
    def __init__(self, game):
        self.nnet = KnotNNet(game)
        self.board_size = len(game.get_canonical_form())
        self.action_size = game.get_action_size()
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=config.learning_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nnet.to(self.device)

        if config.resumeTraining:
            # Load model from 'currentModel/current.pth.tar' if resumeTraining is True
            model_path = 'currentModel/current.pth.tar'
            if os.path.isfile(model_path):
                print(f"Resuming training from {model_path}")
                self.load_checkpoint(model_path)
            else:
                print(f"No model found at {model_path}, starting from scratch.")
        elif config.load_model:
            self.load_checkpoint(os.path.join(config.checkpoint_path, config.load_model_file))

    def train(self, examples):
        self.nnet.train()
        for epoch in range(10):  
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
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.nnet.eval()
        with torch.no_grad():
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
