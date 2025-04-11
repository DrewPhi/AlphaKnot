import os
import numpy as np
from collections import deque
import random
import config
from arena import Arena
from mcts import MCTS
import time
import torch
from multiprocessing import Pool, cpu_count, set_start_method
from knot_graph_game import KnotGraphGame
from knot_graph_nnet import NNetWrapper

# Set multiprocessing start method to avoid CUDA forking issue
set_start_method("spawn", force=True)
def _run_episode_with_coach(coach, ep_num):
    start = time.time()
    result = coach.executeEpisode()
    print(f"[CPU] Self-play Episode {ep_num} completed in {time.time() - start:.2f}s")
    return result
def _init_worker(checkpoint_path, args):
    game = KnotGraphGame()
    game.getInitBoard()
    # Choose CPU or GPU for self-play model based on config
    device = "cpu" if config.use_cpu_in_selfplay or not torch.cuda.is_available() else "cuda"
    nnet = NNetWrapper(game, device=device)
    if checkpoint_path and os.path.isfile(checkpoint_path):
        nnet.load_checkpoint(checkpoint_path)
    return Coach(game, nnet, args)

def run_against_random(game_class, model_path, args, as_first_player):
    from knot_graph_game import KnotGraphGame
    from knot_graph_nnet import NNetWrapper
    from mcts import MCTS
    import torch
    import numpy as np
    import random

    game = game_class()
    game.getInitBoard()

    nnet = NNetWrapper(game, device="cuda" if torch.cuda.is_available() else "cpu")
    nnet.load_checkpoint(model_path)

    def nnet_player(board, player):
        canonicalBoard, current_player = game.getCanonicalForm(board, player)
        pi = MCTS(game, nnet, args).getActionProb(canonicalBoard, current_player, temp=0)
        return np.argmax(pi)

    def random_player(board, player):
        valids = game.getValidMoves(board, player).cpu().numpy()
        valid_actions = np.where(valids == 1)[0]
        return random.choice(valid_actions)

    player1, player2 = (nnet_player, random_player) if as_first_player else (random_player, nnet_player)

    board = game.getInitBoard()
    curPlayer = 1
    while True:
        action = player1(board, curPlayer) if curPlayer == 1 else player2(board, curPlayer)
        board, curPlayer = game.getNextState(board, curPlayer, action)
        result = game.getGameEnded(board, curPlayer)
        if result != 0:
            return int(result * curPlayer)


class PlayerFn:
    def __init__(self, nnet, game, args):
        self.nnet = nnet
        self.game = game
        self.args = args

    def __call__(self, board, player_id):
        canonicalBoard, current_player = self.game.getCanonicalForm(board, player_id)
        pi = MCTS(self.game, self.nnet, self.args).getActionProb(canonicalBoard, current_player, temp=0)
        return np.argmax(pi)



def run_game(game, player1, player2):
    arena = Arena(player1, player2, game)
    return arena.playGame()

class NNetPlayer:
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args

    def __call__(self, board, player):
        canonicalBoard, current_player = self.game.getCanonicalForm(board, player)
        pi = MCTS(self.game, self.nnet, self.args).getActionProb(canonicalBoard, current_player, temp=0)
        return np.argmax(pi)

class RandomPlayer:
    def __init__(self, game):
        self.game = game

    def __call__(self, board, player):
        valids = self.game.getValidMoves(board, player).cpu().numpy()
        valid_actions = np.where(valids == 1)[0]
        return np.random.choice(valid_actions)
    

class Coach:
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args

        self.rank = int(os.environ.get("RANK", "0"))
        self.trainExamplesHistory = deque([], maxlen=config.maxlenOfQueue)

        if config.load_model:
            self.loadTrainExamples()



    def executeEpisode(self):
        trainExamples = []
        board = self.game.getInitBoard()
        curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard, current_player = self.game.getCanonicalForm(board, curPlayer)

            if np.random.rand() < config.random_play_fraction:
                valids = self.game.getValidMoves(canonicalBoard, current_player).cpu().numpy()
                action_probs = valids / np.sum(valids)
            else:
                mcts = MCTS(self.game, self.nnet, self.args, add_root_noise=True)
                action_probs = mcts.getActionProb(canonicalBoard, current_player, temp=1)

            trainExamples.append((canonicalBoard, current_player, action_probs))

            action = np.random.choice(len(action_probs), p=action_probs)
            board, curPlayer = self.game.getNextState(board, curPlayer, action)

            r = self.game.getGameEnded(board, curPlayer)
            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != curPlayer))) for x in trainExamples]





    def evaluate_against_random_model_path(self, model_path, num_games=50):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")

        print("[Arena] Parallel evaluation: AI as Player 1...")
        with ctx.Pool() as pool:
            results_first = pool.starmap(
                run_against_random,
                [(KnotGraphGame, model_path, self.args, True) for _ in range(num_games)]
            )

        print("[Arena] Parallel evaluation: AI as Player 2...")
        with ctx.Pool() as pool:
            results_second = pool.starmap(
                run_against_random,
                [(KnotGraphGame, model_path, self.args, False) for _ in range(num_games)]
            )

        ai_p1_wins = results_first.count(1)
        ai_p2_wins = results_second.count(-1)

        ai_p1_winrate = 100 * ai_p1_wins / num_games
        ai_p2_winrate = 100 * ai_p2_wins / num_games

        print("\nEvaluation vs Random Player:")
        print(f"➡️  AI FIRST:  {ai_p1_winrate:.2f}% wins")
        print(f"⬅️  AI SECOND: {ai_p2_winrate:.2f}% wins")

        return ai_p1_winrate, ai_p2_winrate






    def learn(self):
        total_start = time.time()
        for i in range(1, config.numIters + 1):
            if self.rank == 0:
                print(f'\n{"=" * 30}\n STARTING ITERATION {i}/{config.numIters}\n{"=" * 30}')
            iteration_start = time.time()

            if self.rank == 0:
                # ⚡ Parallel self-play using multiprocessing
                num_workers = min(os.cpu_count(), config.numEps)
                print(f"[CPU] Launching {config.numEps} self-play episodes using {num_workers} workers.")
                checkpoint = os.path.join(config.checkpoint, 'best.pth.tar')

                from multiprocessing import Pool
                with Pool(processes=num_workers) as pool:
                    coaches = [_init_worker(checkpoint, self.args) for _ in range(num_workers)]
                    args = [(coaches[i % num_workers], i + 1) for i in range(config.numEps)]
                    results = pool.starmap(_run_episode_with_coach, args)

                iterationTrainExamples = [item for sublist in results for item in sublist]
                self.trainExamplesHistory.extend(iterationTrainExamples)

                self.saveTrainExamples(i - 1)
                random.shuffle(iterationTrainExamples)
                print("[Training] Started neural network training.")

            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            # Set model to training mode
            if hasattr(self.nnet.model, "module"):
                self.nnet.model.module.train()
            else:
                self.nnet.model.train()

            if torch.distributed.is_initialized():
                # Broadcast examples to all ranks
                data_list = [iterationTrainExamples] if self.rank == 0 else [None]
                torch.distributed.broadcast_object_list(data_list, src=0)
                iterationTrainExamples = data_list[0]

                # Split data for distributed training
                total = len(iterationTrainExamples)
                n_procs = torch.distributed.get_world_size()
                per_rank = total // n_procs
                start = self.rank * per_rank
                end = total if self.rank == n_procs - 1 else (self.rank + 1) * per_rank
                local_examples = iterationTrainExamples[start:end]

                train_start = time.time() if self.rank == 0 else None
                self.nnet.train(local_examples)
                if self.rank == 0:
                    print(f"[Training] Completed distributed training in {time.time() - train_start:.2f}s")
            else:
                train_start = time.time()
                self.nnet.train(iterationTrainExamples)
                print(f"[Training] Completed training in {time.time() - train_start:.2f}s")

            if self.rank == 0:
                if config.saveIterCheckpoint:
                    folder = config.checkpoint
                    filename = f'checkpoint_{i}.pth.tar'
                    self.nnet.save_checkpoint(os.path.join(folder, filename))
                    print(f"[Checkpoint] Saved: {filename}")
                if i == 1:
                    if config.resume_training:
                        print('Evaluating checkpoint 1 against previous champion (best.pth.tar)...')
                        prev_nnet = self.nnet.__class__(self.game)
                        prev_nnet.load_checkpoint(os.path.join(config.checkpoint, 'best.pth.tar'))

                        p1 = PlayerFn(self.nnet, self.game, self.args)
                        p2 = PlayerFn(prev_nnet, self.game, self.args)


                        arena1 = Arena(
                            nnet1_path=os.path.join(config.checkpoint, f'checkpoint_{i}.pth.tar'),
                            nnet2_path=os.path.join(config.checkpoint, 'best.pth.tar'),
                            game_class=self.game.__class__,
                            args=self.args
                        )
                        nwins1, pwins1, draws1 = arena1.playGames_parallel(num_games=config.arenaCompare // 2)

                        arena2 = Arena(
                            nnet1_path=os.path.join(config.checkpoint, 'best.pth.tar'),
                            nnet2_path=os.path.join(config.checkpoint, f'checkpoint_{i}.pth.tar'),
                            game_class=self.game.__class__,
                            args=self.args
                        )
                        pwins2, nwins2, draws2 = arena2.playGames_parallel(num_games=config.arenaCompare // 2)


                        nwins = nwins1 + nwins2
                        pwins = pwins1 + pwins2
                        draws = draws1 + draws2

                        winRate = float(nwins) / (nwins + pwins) if (nwins + pwins) else 0

                        print(f'[Arena Results] New model wins: {nwins}, Previous model wins: {pwins}, Draws: {draws}')
                        print(f'[Arena Results] New model win rate: {winRate:.2%}')

                        if winRate > 0.5:
                            print('New model wins head-to-head. Accepting new model as champion.')
                            self.nnet.save_checkpoint(os.path.join(config.checkpoint, 'best.pth.tar'))
                        elif winRate < 0.5:
                            print('Previous model wins head-to-head. Reverting to previous model.')
                            self.nnet.load_checkpoint(os.path.join(config.checkpoint, f'checkpoint_{i}.pth.tar'))
                        else:
                            print('Tie. Keeping current champion.')
                    else:
                        print('Initial iteration complete; no champion yet — accepting current model as best.')
                        self.nnet.save_checkpoint(os.path.join(config.checkpoint, 'best.pth.tar'))



            if config.arenaCompare > 0 and i > 1 and self.rank == 0:
                print("[Arena] Evaluating against previous model...")
                prev_nnet = self.nnet.__class__(self.game)
                prev_nnet.load_checkpoint(os.path.join(config.checkpoint, 'best.pth.tar'))



                p1 = PlayerFn(self.nnet, self.game, self.args)
                p2 = PlayerFn(prev_nnet, self.game, self.args)


                arena1 = Arena(
                    nnet1_path=os.path.join(config.checkpoint, f'checkpoint_{i}.pth.tar'),
                    nnet2_path=os.path.join(config.checkpoint, 'best.pth.tar'),
                    game_class=self.game.__class__,
                    args=self.args
                )
                nwins1, pwins1, draws1 = arena1.playGames_parallel(num_games=config.arenaCompare // 2)

                arena2 = Arena(
                    nnet1_path=os.path.join(config.checkpoint, 'best.pth.tar'),
                    nnet2_path=os.path.join(config.checkpoint, f'checkpoint_{i}.pth.tar'),
                    game_class=self.game.__class__,
                    args=self.args
                )
                pwins2, nwins2, draws2 = arena2.playGames_parallel(num_games=config.arenaCompare // 2)




                nwins = nwins1 + nwins2
                pwins = pwins1 + pwins2
                draws = draws1 + draws2

                winRate = float(nwins) / (nwins + pwins) if (nwins + pwins) else 0

                print(f'[Arena Results] New model wins: {nwins}, Previous model wins: {pwins}, Draws: {draws}')
                print(f'[Arena Results] New model win rate: {winRate:.2%}')

                if winRate > 0.5:
                    print('New model wins head-to-head. Accepting new model as champion.')
                    self.nnet.save_checkpoint(os.path.join(config.checkpoint, 'best.pth.tar'))
                elif winRate < 0.5:
                    print('Previous model wins head-to-head. Reverting to previous model.')
                    self.nnet.load_checkpoint(os.path.join(config.checkpoint, f'checkpoint_{i - 1}.pth.tar'))
                else:
                    print("Head-to-head tied. Evaluating both models against Random Player...")
                    print("\nEvaluating CURRENT CANDIDATE vs Random Player:")
                    ai_current_p1, ai_current_p2 = self.evaluate_against_random_model_path(
                        os.path.join(config.checkpoint, f"checkpoint_{i}.pth.tar"), num_games=100
                    )

                    print("\nEvaluating PREVIOUS CHAMPION vs Random Player:")
                    ai_prev_p1, ai_prev_p2 = self.evaluate_against_random_model_path(
                        os.path.join(config.checkpoint, "best.pth.tar"), num_games=100
                    )

                    current_avg = (ai_current_p1 + ai_current_p2) / 2
                    prev_avg = (ai_prev_p1 + ai_prev_p2) / 2

                    if current_avg > prev_avg:
                        print('Random evaluation: New model wins. Accepting new model.')
                        self.nnet.save_checkpoint(os.path.join(config.checkpoint, 'best.pth.tar'))
                    elif current_avg < prev_avg:
                        print('Random evaluation: Previous model wins. Reverting to previous model.')
                        self.nnet.load_checkpoint(os.path.join(config.checkpoint, f'checkpoint_{i - 1}.pth.tar'))
                    else:
                        print('Random evaluation tied. Comparing training loss as final tie-breaker.')
                        if self.nnet.latest_loss < getattr(prev_nnet, 'latest_loss', float('inf')):
                            print('New model has lower training loss. Accepting new model.')
                            self.nnet.save_checkpoint(os.path.join(config.checkpoint, 'best.pth.tar'))
                        else:
                            print('Previous model has lower or equal training loss. Reverting to previous model.')
                            self.nnet.load_checkpoint(os.path.join(config.checkpoint, f'checkpoint_{i - 1}.pth.tar'))

            if self.rank == 0:
                iteration_time = (time.time() - iteration_start) / 60
                total_elapsed = (time.time() - total_start) / 60
                est_remaining = (total_elapsed / i) * (config.numIters - i)
                print(f"[Iteration] Iteration time: {iteration_time:.2f} minutes | "
                    f"Total elapsed: {total_elapsed:.2f} minutes | "
                    f"Estimated remaining: {est_remaining:.2f} minutes")







    def saveTrainExamples(self, iteration):
        folder = config.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, f'train_examples_{iteration}.pkl')
        with open(filename, "wb") as f:
            import pickle
            pickle.dump(self.trainExamplesHistory, f)

    def loadTrainExamples(self):
        filename = os.path.join(config.load_folder_file[0], config.load_folder_file[1]+".examples")
        if not os.path.isfile(filename):
            print(f"File {filename} with trainExamples not found!")
        else:
            print("File with trainExamples found. Loading...")
            with open(filename, "rb") as f:
                import pickle
                self.trainExamplesHistory = pickle.load(f)
            print('Loading done!')