# coach.py

import random
import numpy as np
import time
import os
import config
from mcts import MCTS
from game import KnotGame
from neural_network import NNetWrapper
import torch
import copy

class Coach:
    def __init__(self, game, nnet):
        self.game = game
        self.nnet = nnet
        self.num_mcts_sims = config.num_mcts_sims
        self.num_iterations = config.num_iterations
        self.num_episodes = config.num_episodes
        self.save_model_freq = config.save_model_freq
        self.checkpoint_path = config.checkpoint_path

        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

    def execute_episode(self):
        train_examples = []
        starting_player = 1 if config.knotter_first else -1
        game = KnotGame(self.game.initial_pd_code, starting_player)
        self.mcts = MCTS(game, self.nnet)

        while not game.game_over:
            canonicalBoard = game.get_canonical_form()
            temp = 1 if len(train_examples) < 10 else 0

            for _ in range(self.num_mcts_sims):
                self.mcts.search(game.clone(), is_root=True)

            pi = self.get_policy(canonicalBoard, temp)
            train_examples.append([canonicalBoard, pi, game.get_current_player()])
            action = np.random.choice(len(pi), p=pi)
            game.make_move(action)

        winner = game.winner
        return [
            (x[0], x[1], winner if x[2] == game.starting_player else -winner)
            for x in train_examples
        ]


    def get_policy(self, canonicalBoard, temp):
        s = self._hash_state(canonicalBoard)
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

        champion_dir = 'championModel'
        if not os.path.exists(champion_dir):
            os.makedirs(champion_dir)
        champion_path = os.path.join(champion_dir, 'best.pth.tar')

        if not os.path.isfile(champion_path):
            self.nnet.save_checkpoint(champion_path)
            print("No champion model found. Initializing current net as champion.")

        i = 1
        champion_perfected = False
        while True:
            iteration_start_time = time.time()
            examples = []

            for e in range(1, self.num_episodes + 1):
                ep_start = time.time()
                episode_data = self.execute_episode()
                examples.extend(episode_data)
                print(f"Iteration {i}/{self.num_iterations}, Episode {e}/{self.num_episodes} "
                    f"completed in {time.time()-ep_start:.2f} s.")

            self.nnet.train(examples)

            iteration_time = time.time() - iteration_start_time
            elapsed_time = time.time() - total_start_time

            if i <= self.num_iterations:
                iterations_left = self.num_iterations - i
                est_time_left = (elapsed_time / i) * iterations_left
                print(f"Iteration {i} completed in {iteration_time/60:.2f} min.")
                print(f"Estimated {est_time_left/60:.2f} min left.")
            else:
                print(f"Extra Iteration {i} completed in {iteration_time/60:.2f} min. "
                    f"(Continuing because require_perfect_random_win=True and champion not perfected yet.)")

            if i % self.save_model_freq == 0 or i == self.num_iterations:
                ckpt_file = os.path.join(self.checkpoint_path, f'checkpoint_{i}.pth.tar')
                self.nnet.save_checkpoint(ckpt_file)
                print(f"Model checkpoint saved to {ckpt_file}.")

            self.compare_and_promote(i)

            if config.require_perfect_random_win:
                wins_first, wins_second = self.evaluate_model_vs_random(self.nnet, 200)
                if wins_first == 200 or wins_second == 200:
                    champion_perfected = True
                    role = "going first" if wins_first == 200 else "going second"
                    print(f"Champion achieved perfect play {role} (200/200 wins) at iteration {i}.")
                else:
                    print(f"Champion wins: {wins_first}/200 when going first, {wins_second}/200 when going second. Still not perfected.")
            else:
                champion_perfected = True

            if i >= self.num_iterations:
                if not config.require_perfect_random_win:
                    break
                if config.require_perfect_random_win and champion_perfected:
                    break
                if config.require_perfect_random_win and not champion_perfected:
                    if i == self.num_iterations:
                        print("Planned iterations completed, but champion hasn't perfected vs random. Extending training.")
            i += 1

        print("Training complete.")


    def compare_and_promote(self, iteration_idx):
        """
        Head-to-head match: candidate vs champion.
        """
        candidate_dir = 'candidateModels'
        if not os.path.exists(candidate_dir):
            os.makedirs(candidate_dir)
        candidate_path = os.path.join(candidate_dir, f'candidate_iter_{iteration_idx}.pth.tar')
        self.nnet.save_checkpoint(candidate_path)
        print(f"Iteration {iteration_idx}: Candidate model saved to {candidate_path}.")

        # Load champion model
        champion_dir = 'championModel'
        champion_path = os.path.join(champion_dir, 'best.pth.tar')
        from neural_network import NNetWrapper
        champion_net = NNetWrapper(self.game)
        champion_net.load_checkpoint(champion_path)

        # Head-to-head match
        champ_wins, cand_wins = self.evaluate_head_to_head(champion_net, self.nnet, config.n_games)
        print(f"Head-to-head results (champ vs candidate): {champ_wins}-{cand_wins} out of {config.n_games} games.")

        # Tie case: compare composite scores against random player
        if cand_wins == champ_wins:
            cand_first, cand_second = self.evaluate_model_vs_random(self.nnet, 100)
            champ_first, champ_second = self.evaluate_model_vs_random(champion_net, 100)

            def composite(f, s):
                max_win = max(f, s)
                avg_win = (f + s) / 2
                stability = 1 - abs(f - s) / 100
                power = max_win / 100
                alpha = 1 - stability * power
                beta = stability * power
                return alpha * max_win + beta * avg_win

            cand_score = composite(cand_first, cand_second)
            champ_score = composite(champ_first, champ_second)

            print(f"Tie-breaker: Composite scores vs random — Candidate: {cand_score:.2f}, Champion: {champ_score:.2f}")

            if cand_score > champ_score:
                self.nnet.save_checkpoint(champion_path)
                print("Candidate outperformed champion in composite score. Promoted to champion.")
            else:
                self.nnet.load_checkpoint(champion_path)
                print("Champion remains stronger. Reverting to champion model.")
        elif cand_wins > champ_wins:
            self.nnet.save_checkpoint(champion_path)
            print("Candidate outperformed champion. Promoted to champion.")
        else:
            self.nnet.load_checkpoint(champion_path)
            print("Champion remains stronger. Reverting to champion model.")


    def evaluate_head_to_head(self, champion_nnet, candidate_nnet, n_games):
        """
        Play n_games between champion and candidate, alternating the starting player.
        Returns (champion wins, candidate wins).
        """
        from game import KnotGame
        champ_wins = 0
        cand_wins = 0
        pd_code = self.game.initial_pd_code

        for game_idx in range(n_games):
            # Alternate starting: even-indexed games: champion starts, odd-indexed: candidate starts.
            if game_idx % 2 == 0:
                champ_player, cand_player = 1, -1
            else:
                champ_player, cand_player = -1, 1

            start_player = 1  # Always set game to start with Player 1
            game = KnotGame(pd_code, start_player)

            while not game.game_over:
                current_pl = game.get_current_player()
                if current_pl == champ_player:
                    state = game.get_canonical_form()
                    pi, _ = champion_nnet.predict(state)
                    valid_moves = game.get_valid_moves()
                    pi = pi * valid_moves
                    pi = pi / np.sum(pi) if np.sum(pi) > 1e-12 else valid_moves / np.sum(valid_moves)
                    action = np.argmax(pi)
                else:
                    state = game.get_canonical_form()
                    pi, _ = candidate_nnet.predict(state)
                    valid_moves = game.get_valid_moves()
                    pi = pi * valid_moves
                    pi = pi / np.sum(pi) if np.sum(pi) > 1e-12 else valid_moves / np.sum(valid_moves)
                    action = np.argmax(pi)
                game.make_move(action)

            if game.winner == champ_player:
                champ_wins += 1
            elif game.winner == cand_player:
                cand_wins += 1

        return champ_wins, cand_wins

    def evaluate_model_vs_random(self, nnet, n_eval_games=200):
        """
        Evaluate the given net vs a random player.
        Plays n_eval_games with AI going first and n_eval_games with AI going second.
        Returns a tuple: (wins_when_AI_first, wins_when_AI_second)
        """
        import random
        from game import KnotGame

        pd_code = self.game.initial_pd_code
        ai_role = 1 if config.knotter_first else -1

        wins_first = 0
        wins_second = 0

        # Evaluate when AI goes first
        for _ in range(n_eval_games):
            game = KnotGame(pd_code, ai_role)
            while not game.game_over:
                cp = game.get_current_player()
                if cp == ai_role:
                    state = game.get_canonical_form()
                    pi, _ = nnet.predict(state)
                    valid_moves = game.get_valid_moves()
                    pi = pi * valid_moves
                    pi = pi / np.sum(pi) if np.sum(pi) > 1e-12 else valid_moves / np.sum(valid_moves)
                    action = np.argmax(pi)
                else:
                    valid_moves = game.get_valid_moves()
                    valid_actions = np.where(valid_moves == 1)[0]
                    action = random.choice(valid_actions)
                game.make_move(action)
            if game.winner == ai_role:
                wins_first += 1

        # Evaluate when AI goes second
        random_start = -ai_role
        for _ in range(n_eval_games):
            game = KnotGame(pd_code, random_start)
            while not game.game_over:
                cp = game.get_current_player()
                if cp == ai_role:
                    state = game.get_canonical_form()
                    pi, _ = nnet.predict(state)
                    valid_moves = game.get_valid_moves()
                    pi = pi * valid_moves
                    pi = pi / np.sum(pi) if np.sum(pi) > 1e-12 else valid_moves / np.sum(valid_moves)
                    action = np.argmax(pi)
                else:
                    valid_moves = game.get_valid_moves()
                    valid_actions = np.where(valid_moves == 1)[0]
                    action = random.choice(valid_actions)
                game.make_move(action)
            if game.winner == ai_role:
                wins_second += 1

        return wins_first, wins_second

    def _hash_state(self, state):
        return state.tobytes()
