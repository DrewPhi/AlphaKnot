import config
from knot_graph_game import KnotGraphGame
from knot_graph_nnet import NNetWrapper
from coach import Coach
from arena import Arena
import numpy as np
from mcts import MCTS
import os, torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import multiprocessing as mp

class Args:
    numMCTSSims = config.numMCTSSims
    cpuct = config.cpuct

def currentNetPlayer(game, nnet, board, player):
    canonicalBoard, current_player = game.getCanonicalForm(board, player)
    pi = MCTS(game, nnet, config, add_root_noise=False).getActionProb(canonicalBoard, current_player, temp=0)
    return np.argmax(pi)


def main():
    # Determine if running in distributed mode
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://')
        rank = dist.get_rank()                # global rank of this process
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        torch.cuda.set_device(local_rank)     # use GPU corresponding to this process
        print(f"[GPU {rank}] Starting on device: {torch.cuda.get_device_name(local_rank)} "
          f"({torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.2f} GB)")
    else:
        rank = 0
        local_rank = 0
        if torch.cuda.is_available():
            print(f"[GPU 0] Single-GPU mode: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)")
        else:
            print("[CPU] CUDA not available, using CPU mode")
    game = KnotGraphGame()
    game.getInitBoard()
    nnet = NNetWrapper(game)
    # Move model to this process's device and wrap with DDP if using multi-GPU
    if world_size > 1:
        nnet.model.to(f'cuda:{local_rank}')
        nnet.model = DDP(nnet.nnet, device_ids=[local_rank], output_device=local_rank)
    coach = Coach(game, nnet, Args())

    coach.learn()   # perform training (possibly distributed)

    if rank == 0:
        # (Optional) Only rank 0 performs final evaluation or prints results
        arena = Arena(lambda b,p: currentNetPlayer(game, nnet, b, p),
                      lambda b,p: currentNetPlayer(game, nnet, b, p), game)
        results = arena.playGames(num=config.arenaCompare, verbose=False)
        print(f"Arena results: {results}")
    # Clean up distributed group
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()