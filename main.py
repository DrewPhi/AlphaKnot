import config
from knot_graph_game import KnotGraphGame
from knot_graph_nnet import NNetWrapper
from coach import Coach
import os, torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import multiprocessing as mp

class Args:
    numMCTSSims = config.numMCTSSims
    cpuct = config.cpuct

def main():
    config.validate()
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
    if config.resume_training:
        champion_path = os.path.join(config.checkpoint, 'best.pth.tar')
        if not os.path.isfile(champion_path):
            raise FileNotFoundError(
                f"resume_training=True but champion is missing: {champion_path}"
            )
        nnet.load_checkpoint(champion_path)
    # Move model to this process's device and wrap with DDP if using multi-GPU
    if world_size > 1:
        nnet.model.to(f'cuda:{local_rank}')
        nnet.model = DDP(nnet.model, device_ids=[local_rank], output_device=local_rank)
    coach = Coach(game, nnet, Args())

    coach.learn()   # perform training (possibly distributed)

    if rank == 0:
        print("Training completed; champion saved to "
              f"{os.path.join(config.checkpoint, 'best.pth.tar')}")
    # Clean up distributed group
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
