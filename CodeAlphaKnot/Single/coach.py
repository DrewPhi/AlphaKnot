
import random
import numpy as np
import time
import os
import config 
from mcts import MCTS
from game import KnotGame  

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

        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

    def execute_episode(self):
        train_examples = []
        starting_player = -1 if config.knotter_first else 1
        game = KnotGame(self.game.initial_pd_code, starting_player)
        while not game.game_over:
            canonicalBoard = game.get_canonical_form()
            temp = 1
            for _ in range(self.num_mcts_sims):
                self.mcts.search(game.clone())
            pi = self.get_policy(canonicalBoard, temp)
            train_examples.append([canonicalBoard, pi, game.get_current_player()])
            action = np.random.choice(len(pi), p=pi)
            game.make_move(action)
        winner = game.winner
        return [(x[0], x[1], winner if x[2] == game.starting_player else -winner) for x in train_examples]

    def get_policy(self, canonicalBoard, temp):
        s = self.mcts._hash_state(canonicalBoard)
        counts = [self.mcts.Nsa.get((s, a), 0) for a in range(self.game.get_action_size())]
        if temp == 0:
            best_as = np.array(np.argwhere(counts == np.max(counts))).flatten()
            probs = np.zeros(len(counts))
            probs[random.choice(best_as)] = 1.0
            return probs
        counts = np.array(counts, dtype=np.float32)
        counts = counts ** (1. / temp)
        counts_sum = float(np.sum(counts))
        probs = counts / counts_sum if counts_sum > 0 else counts
        return probs

    def learn(self):
        total_start_time = time.time()
        for i in range(1, self.num_iterations + 1):
            iteration_start_time = time.time()
            self.examples = []
            for e in range(1, self.num_episodes + 1):
                episode_start_time = time.time()
                self.examples += self.execute_episode()
                episode_time = time.time() - episode_start_time
                print(f"Iteration {i}/{self.num_iterations}, Episode {e}/{self.num_episodes} completed in {episode_time:.2f} seconds.")
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
