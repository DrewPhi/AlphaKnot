# coach_multi.py

import random
import numpy as np
import time
import os
import config  # Import config.py
from mcts_multi import MCTS
from game_multi import KnotGame  # Import KnotGame
from multiprocessing import Pool, cpu_count
import torch

class Coach:
    def __init__(self, game, nnet):
        self.game = game
        self.nnet = nnet
        self.mcts = MCTS(game, nnet)
        self.examples = []
        self.num_mcts_sims = config.num_mcts_sims
        self.num_iterations = config.num_iterations
        self.num_episodes = config.num_episodes
        self.save_model_freq = config.save_model_freq
        self.checkpoint_path = config.checkpoint_path
        self.num_parallel_games = config.num_parallel_games

        # Create the checkpoint directory if it doesn't exist
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        # Check and log GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {torch.cuda.device_count()} GPUs on device {self.device}")

    def execute_episode(self, _=None):
        """ Plays one self-play episode using MCTS. """
        train_examples = []
        starting_player = -1 if config.knotter_first else 1
        game = KnotGame(self.game.pd_codes, starting_player)
        mcts = MCTS(game, self.nnet)  # Each episode gets its own MCTS instance

        while not game.game_over:
            canonicalBoard = game.get_canonical_form()
            temp = 1
            for _ in range(self.num_mcts_sims):
                mcts.search(game.clone())
            pi = self.get_policy(mcts, canonicalBoard, temp)
            train_examples.append([canonicalBoard, pi, game.get_current_player()])
            action = np.random.choice(len(pi), p=pi)
            game.make_move(action)

        winner = game.winner
        return [(x[0], x[1], winner if x[2] == game.starting_player else -winner) for x in train_examples]

    def get_policy(self, mcts, canonicalBoard, temp):
        """ Computes the move probability distribution using MCTS visit counts. """
        s = mcts._hash_state(canonicalBoard)
        counts = [mcts.Nsa.get((s, a), 0) for a in range(self.game.get_action_size())]

        if temp == 0:
            best_as = np.array(np.argwhere(counts == np.max(counts))).flatten()
            probs = np.zeros(len(counts))
            probs[random.choice(best_as)] = 1.0
            return probs

        counts = np.array(counts, dtype=np.float32)
        counts = counts ** (1. / temp)
        counts_sum = float(np.sum(counts))
        return counts / counts_sum if counts_sum > 0 else counts

    def learn(self):
        """ Runs self-play, training, and model saving. """
        total_start_time = time.time()

        for i in range(1, self.num_iterations + 1):
            iteration_start_time = time.time()
            self.examples = []

            print(f"Iteration {i}/{self.num_iterations}: Running {self.num_episodes} self-play episodes...")

            with Pool(self.num_parallel_games) as pool:
                all_examples = pool.map(self.execute_episode, range(self.num_episodes))
            
            # Flatten the list of examples
            self.examples = [item for sublist in all_examples for item in sublist]

            print(f"Iteration {i}: Training the neural network on {len(self.examples)} examples...")
            self.nnet.train(self.examples)

            iteration_time = time.time() - iteration_start_time
            elapsed_time = time.time() - total_start_time
            iterations_left = self.num_iterations - i
            estimated_time_left = (elapsed_time / i) * iterations_left

            print(f"Iteration {i} completed in {iteration_time / 60:.2f} minutes.")
            print(f"Estimated time left: {estimated_time_left / 60:.2f} minutes.")

            if i % self.save_model_freq == 0 or i == self.num_iterations:
                model_filename = os.path.join(self.checkpoint_path, f'checkpoint_{i}.pth.tar')
                self.nnet.save_checkpoint(model_filename)
                print(f"Model checkpoint saved to {model_filename}.")
