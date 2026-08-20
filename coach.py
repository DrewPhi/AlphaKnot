import os
import numpy as np
from collections import deque
import random
import config
from arena import Arena
from mcts import MCTS
import time
import torch
import math
from multiprocessing import get_context
from knot_graph_game import KnotGraphGame
from knot_graph_nnet import NNetWrapper
from pd_code_utils import serialize_graph

_WORKER_COACH = None
_RANDOM_EVAL_STATE = None


def allocated_cpu_count():
    """Return the scheduler allocation instead of the host's total CPUs."""
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    configured = getattr(config, "selfplay_workers", 0)
    return min(allocated, configured) if configured > 0 else allocated


def equal_ddp_shard(examples, rank, world_size):
    """Return a padded shard so every DDP rank runs equal backward steps."""
    if not examples:
        raise ValueError("examples must not be empty")
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid DDP rank/world_size")
    per_rank = math.ceil(len(examples) / world_size)
    padded = list(examples)
    needed = per_rank * world_size - len(padded)
    if needed:
        repeats = math.ceil(needed / len(examples))
        padded.extend((list(examples) * repeats)[:needed])
    start = rank * per_rank
    return padded[start:start + per_rank]


def _run_episode(ep_num):
    start = time.time()
    result = _WORKER_COACH.executeEpisode()
    print(f"[CPU] Self-play Episode {ep_num} completed in {time.time() - start:.2f}s")
    return result


def _init_worker(checkpoint_path, args):
    global _WORKER_COACH
    game = KnotGraphGame()
    game.getInitBoard()
    # One CUDA model per process quickly exhausts device memory.  Production
    # self-play defaults to CPU workers; GPUs are reserved for DDP training.
    device = getattr(config, "selfplay_device", "cpu")
    if device != "cpu" and not torch.cuda.is_available():
        device = "cpu"
    nnet = NNetWrapper(game, device=device)
    nnet.load_checkpoint(checkpoint_path, load_optimizer=False)
    _WORKER_COACH = Coach(game, nnet, args, load_history=False)

def _init_random_eval_worker(game_class, model_path, args, as_first_player):
    global _RANDOM_EVAL_STATE
    game = game_class()
    game.getInitBoard()
    nnet = NNetWrapper(game, device="cpu")
    nnet.load_checkpoint(model_path, load_optimizer=False)
    _RANDOM_EVAL_STATE = (game, nnet, args, as_first_player)


def _run_random_game(_game_number):
    game, nnet, args, as_first_player = _RANDOM_EVAL_STATE

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


class Coach:
    def __init__(self, game, nnet, args, load_history=True):
        self.game = game
        self.nnet = nnet
        self.args = args

        self.rank = int(os.environ.get("RANK", "0"))
        self.trainExamplesHistory = deque([], maxlen=config.maxlenOfQueue)

        if load_history and config.load_model:
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
                return [
                    (
                        serialize_graph(example_board),
                        policy,
                        r * ((-1) ** (example_player != curPlayer)),
                    )
                    for example_board, example_player, policy in trainExamples
                ]





    def evaluate_against_random_model_path(self, model_path, num_games=50):
        ctx = get_context("spawn")

        print("[Arena] Parallel evaluation: AI as Player 1...")
        num_workers = min(allocated_cpu_count(), num_games)
        with ctx.Pool(
            processes=num_workers,
            initializer=_init_random_eval_worker,
            initargs=(KnotGraphGame, model_path, self.args, True),
        ) as pool:
            results_first = pool.map(_run_random_game, range(num_games))

        print("[Arena] Parallel evaluation: AI as Player 2...")
        with ctx.Pool(
            processes=num_workers,
            initializer=_init_random_eval_worker,
            initargs=(KnotGraphGame, model_path, self.args, False),
        ) as pool:
            results_second = pool.map(_run_random_game, range(num_games))

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
        best_path = os.path.join(config.checkpoint, 'best.pth.tar')
        if self.rank == 0:
            if config.resume_training and not os.path.isfile(best_path):
                raise FileNotFoundError(
                    f"resume_training=True but champion is missing: {best_path}"
                )
            if not config.resume_training:
                # Self-play for the first iteration still needs a checkpoint.
                # This is the initial random champion and is replaced after
                # the first training pass.
                self.nnet.save_checkpoint(best_path)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        for i in range(1, config.numIters + 1):
            if self.rank == 0:
                print(f'\n{"=" * 30}\n STARTING ITERATION {i}/{config.numIters}\n{"=" * 30}')
            iteration_start = time.time()

            if self.rank == 0:
                # ⚡ Parallel self-play using multiprocessing
                num_workers = min(allocated_cpu_count(), config.numEps)
                print(f"[CPU] Launching {config.numEps} self-play episodes using {num_workers} workers.")
                checkpoint = os.path.join(config.checkpoint, 'best.pth.tar')

                context = get_context("spawn")
                with context.Pool(
                    processes=num_workers,
                    initializer=_init_worker,
                    initargs=(checkpoint, self.args),
                ) as pool:
                    results = pool.map(_run_episode, range(1, config.numEps + 1))

                iterationTrainExamples = [item for sublist in results for item in sublist]
                self.trainExamplesHistory.extend(iterationTrainExamples)

                self.saveTrainExamples(i - 1)
                iterationTrainExamples = list(self.trainExamplesHistory)
                random.Random(i).shuffle(iterationTrainExamples)
                print("[Training] Started neural network training.")

            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            # Set model to training mode
            if hasattr(self.nnet.model, "module"):
                self.nnet.model.module.train()
            else:
                self.nnet.model.train()

            if torch.distributed.is_initialized():
                # Avoid serializing the full history through NCCL/GPU memory.
                # All ranks on the single-node production job read the shared
                # checkpoint file and apply the same deterministic shuffle.
                examples_path = os.path.join(
                    config.checkpoint, f'train_examples_{i - 1}.pkl'
                )
                with open(examples_path, "rb") as examples_file:
                    import pickle
                    iterationTrainExamples = list(pickle.load(examples_file))
                random.Random(i).shuffle(iterationTrainExamples)

                # Split data into equal-length shards. DDP performs a gradient
                # synchronization for every backward pass, so unequal shard
                # lengths can deadlock ranks on the final steps.
                total = len(iterationTrainExamples)
                n_procs = torch.distributed.get_world_size()
                if total == 0:
                    raise RuntimeError("Self-play produced no training examples")
                local_examples = equal_ddp_shard(
                    iterationTrainExamples, self.rank, n_procs
                )

                train_start = time.time() if self.rank == 0 else None
                self.nnet.train(local_examples)
                if self.rank == 0:
                    print(f"[Training] Completed distributed training in {time.time() - train_start:.2f}s")
            else:
                train_start = time.time()
                self.nnet.train(iterationTrainExamples)
                print(f"[Training] Completed training in {time.time() - train_start:.2f}s")

            if self.rank == 0:
                if config.saveIterCheckpoint or config.arenaCompare > 0:
                    folder = config.checkpoint
                    filename = f'checkpoint_{i}.pth.tar'
                    self.nnet.save_checkpoint(os.path.join(folder, filename))
                    print(f"[Checkpoint] Saved: {filename}")
                if i == 1:
                    if config.resume_training:
                        print('Evaluating checkpoint 1 against previous champion (best.pth.tar)...')
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
                            self.nnet.load_checkpoint(os.path.join(config.checkpoint, 'best.pth.tar'))
                        else:
                            print('Tie. Keeping current champion.')
                            self.nnet.load_checkpoint(os.path.join(config.checkpoint, 'best.pth.tar'))
                    else:
                        print('Initial iteration complete; no champion yet — accepting current model as best.')
                        self.nnet.save_checkpoint(os.path.join(config.checkpoint, 'best.pth.tar'))



            if config.arenaCompare > 0 and i > 1 and self.rank == 0:
                print("[Arena] Evaluating against previous model...")
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
                    self.nnet.load_checkpoint(os.path.join(config.checkpoint, 'best.pth.tar'))
                else:
                    print("Head-to-head tied. Evaluating both models against Random Player...")
                    print("\nEvaluating CURRENT CANDIDATE vs Random Player:")
                    ai_current_p1, ai_current_p2 = self.evaluate_against_random_model_path(
                        os.path.join(config.checkpoint, f"checkpoint_{i}.pth.tar"), num_games=config.randomGames
                    )

                    print("\nEvaluating PREVIOUS CHAMPION vs Random Player:")
                    ai_prev_p1, ai_prev_p2 = self.evaluate_against_random_model_path(
                        os.path.join(config.checkpoint, "best.pth.tar"), num_games=config.randomGames
                    )

                    current_avg = (ai_current_p1 + ai_current_p2) / 2
                    prev_avg = (ai_prev_p1 + ai_prev_p2) / 2

                    if current_avg > prev_avg:
                        print('Random evaluation: New model wins. Accepting new model.')
                        self.nnet.save_checkpoint(os.path.join(config.checkpoint, 'best.pth.tar'))
                    elif current_avg < prev_avg:
                        print('Random evaluation: Previous model wins. Reverting to previous model.')
                        self.nnet.load_checkpoint(os.path.join(config.checkpoint, 'best.pth.tar'))
                    else:
                        print('Random evaluation tied. Keeping previous champion.')
                        self.nnet.load_checkpoint(os.path.join(config.checkpoint, 'best.pth.tar'))

            if config.arenaCompare <= 0 and i > 1 and self.rank == 0:
                print('[Arena] Disabled; accepting candidate as champion.')
                self.nnet.save_checkpoint(best_path)

            # Every DDP rank must begin the next iteration from the exact same
            # accepted champion, including after a rank-0 arena rejection.
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            self.nnet.load_checkpoint(best_path)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

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
        filename = config.train_examples_path
        if not filename:
            raise ValueError(
                "load_model=True requires config.train_examples_path"
            )
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"Training examples not found: {filename}")
        else:
            print("File with trainExamples found. Loading...")
            with open(filename, "rb") as f:
                import pickle
                self.trainExamplesHistory = pickle.load(f)
            print('Loading done!')
