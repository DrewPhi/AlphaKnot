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
            nn_input = game.get_nn_input()  # Use full generalized PD code
            temp = 1 if len(train_examples) < 10 else 0

            for _ in range(self.num_mcts_sims):
                self.mcts.search(game.clone(), is_root=True)

            pi_crossing, pi_res = self.get_policy(nn_input, temp)
            train_examples.append([nn_input, (pi_crossing, pi_res), game.get_current_player()])

            # Create joint distribution over actions (crossing * resolution)
            joint_policy = np.outer(pi_crossing, pi_res).flatten()
            valid_moves = game.get_valid_moves()
            joint_policy *= valid_moves
            joint_policy /= np.sum(joint_policy) if np.sum(joint_policy) > 1e-12 else 1

            action = np.random.choice(len(joint_policy), p=joint_policy)
            game.make_move(action)

        winner = game.winner
        return [
            (x[0], x[1], winner if x[2] == game.starting_player else -winner)
            for x in train_examples
        ]






    def get_policy(self, nn_input, temp):
        s = self._hash_state(nn_input)
        num_crossings = len(nn_input)

        counts = [self.mcts.Nsa.get((s, a), 0) for a in range(num_crossings * 2)]
        counts = np.array(counts, dtype=np.float32).reshape(num_crossings, 2)

        if temp == 0:
            max_idx = np.unravel_index(np.argmax(counts), counts.shape)
            pi_crossing = np.zeros(num_crossings, dtype=np.float32)
            pi_res = np.zeros(2, dtype=np.float32)
            pi_crossing[max_idx[0]] = 1.0
            pi_res[max_idx[1]] = 1.0
            return pi_crossing, pi_res

        # Apply temperature
        counts = counts ** (1. / temp)
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # prevent division by zero

        pi = counts / row_sums

        pi_crossing = row_sums.flatten()
        pi_crossing /= np.sum(pi_crossing) if np.sum(pi_crossing) > 0 else 1

        pi_res = pi.mean(axis=0)
        pi_res /= np.sum(pi_res) if np.sum(pi_res) > 0 else 1

        return pi_crossing, pi_res





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

        if config.promote_by_loss_only:
            champ_net_loss = self.load_champion_loss()
            cand_loss = self.nnet.last_training_loss
            print(f"Loss comparison — Candidate: {cand_loss:.4f}, Champion: {champ_net_loss:.4f}" if champ_net_loss is not None else f"Loss comparison — Candidate: {cand_loss:.4f}, Champion: None")

            if champ_net_loss is None or cand_loss < champ_net_loss:
                self.nnet.save_checkpoint(champion_path)
                self.save_champion_loss(cand_loss)
                print("Candidate has lower training loss. Promoted to champion (loss-based).")
            else:
                print("Candidate has higher training loss. Continuing to improve it.")
            return

        champ_wins, cand_wins = self.evaluate_head_to_head(champion_net, self.nnet, config.n_games)
        print(f"Head-to-head results (champ vs candidate): {champ_wins}-{cand_wins} out of {config.n_games} games.")

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
                self.save_champion_loss(self.nnet.last_training_loss)
                print("Candidate outperformed champion in composite score. Promoted to champion.")
            else:
                print("Candidate tied but underperformed in composite score. Continuing to improve it.")
        elif cand_wins > champ_wins:
            self.nnet.save_checkpoint(champion_path)
            self.save_champion_loss(self.nnet.last_training_loss)
            print("Candidate outperformed champion. Promoted to champion.")
        else:
            print("Candidate failed to outperform champion. Continuing to improve it.")




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
            if game_idx % 2 == 0:
                champ_player, cand_player = 1, -1
            else:
                champ_player, cand_player = -1, 1

            game = KnotGame(pd_code, start_player=1)

            while not game.game_over:
                cp = game.get_current_player()
                state = game.get_nn_input()

                if cp == champ_player:
                    (pi_cross, pi_res), _ = champion_nnet.predict(state)
                else:
                    (pi_cross, pi_res), _ = candidate_nnet.predict(state)

                joint_pi = np.outer(pi_cross, pi_res).flatten()
                valid_moves = game.get_valid_moves()
                joint_pi *= valid_moves
                joint_pi /= np.sum(joint_pi) if np.sum(joint_pi) > 1e-12 else 1
                action = np.argmax(joint_pi)

                game.make_move(action)

            if game.winner == champ_player:
                champ_wins += 1
            elif game.winner == cand_player:
                cand_wins += 1

        return champ_wins, cand_wins

    def save_champion_loss(self, loss_value):
        with open(os.path.join('championModel', 'loss.txt'), 'w') as f:
            f.write(str(loss_value))

    def load_champion_loss(self):
        try:
            with open(os.path.join('championModel', 'loss.txt'), 'r') as f:
                return float(f.read().strip())
        except:
            return None

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

        # AI goes first
        for _ in range(n_eval_games):
            game = KnotGame(pd_code, ai_role)
            while not game.game_over:
                cp = game.get_current_player()
                state = game.get_nn_input()
                if cp == ai_role:
                    pi_cross, pi_res, value = nnet.predict(state)
                    joint_pi = np.outer(pi_cross, pi_res).flatten()
                    valid_moves = game.get_valid_moves()
                    joint_pi *= valid_moves
                    joint_pi /= np.sum(joint_pi) if np.sum(joint_pi) > 1e-12 else 1
                    action = np.argmax(joint_pi)
                else:
                    valid_actions = np.where(game.get_valid_moves() == 1)[0]
                    action = random.choice(valid_actions)
                game.make_move(action)
            if game.winner == ai_role:
                wins_first += 1

        # AI goes second
        random_start = -ai_role
        for _ in range(n_eval_games):
            game = KnotGame(pd_code, random_start)
            while not game.game_over:
                cp = game.get_current_player()
                state = game.get_nn_input()
                if cp == ai_role:
                    pi_cross, pi_res, value = nnet.predict(state)
                    joint_pi = np.outer(pi_cross, pi_res).flatten()
                    valid_moves = game.get_valid_moves()
                    joint_pi *= valid_moves
                    joint_pi /= np.sum(joint_pi) if np.sum(joint_pi) > 1e-12 else 1
                    action = np.argmax(joint_pi)
                else:
                    valid_actions = np.where(game.get_valid_moves() == 1)[0]
                    action = random.choice(valid_actions)
                game.make_move(action)
            if game.winner == ai_role:
                wins_second += 1

        return wins_first, wins_second


    def _hash_state(self, state):
        return state.tobytes()
