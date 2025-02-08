# main_multi.py

from game_multi import KnotGame
from neural_network_multi import NNetWrapper
from coach_multi import Coach
import time
import os
import config  # Import config.py
import torch
print(f"Using {torch.cuda.device_count()} GPUs.")

# Your provided general PD codes for multiple knots
pd_codes = [
    # '6_1' knot
    [[[1, 7, 2, 6, 0], [6, 1, 7, 2, 0]],
     [[3, 10, 4, 11, 0], [10, 4, 11, 3, 0]],
     [[5, 3, 6, 2, 0], [2, 5, 3, 6, 0]],
     [[7, 1, 8, 12, 0], [1, 8, 12, 7, 0]],
     [[9, 4, 10, 5, 0], [4, 10, 5, 9, 0]],
     [[11, 9, 12, 8, 0], [8, 11, 9, 12, 0]]],
    # '6_2' knot
    [[[1, 8, 2, 9, 0], [8, 2, 9, 1, 0]],
     [[3, 11, 4, 10, 0], [10, 3, 11, 4, 0]],
     [[5, 1, 6, 12, 0], [1, 6, 12, 5, 0]],
     [[7, 2, 8, 3, 0], [2, 8, 3, 7, 0]],
     [[9, 7, 10, 6, 0], [6, 9, 7, 10, 0]],
     [[11, 5, 12, 4, 0], [4, 11, 5, 12, 0]]],
    # '6_3' knot
    [[[4, 2, 5, 1, 0], [1, 4, 2, 5, 0]],
     [[8, 4, 9, 3, 0], [3, 8, 4, 9, 0]],
     [[12, 9, 1, 10, 0], [9, 1, 10, 12, 0]],
     [[10, 5, 11, 6, 0], [5, 11, 6, 10, 0]],
     [[6, 11, 7, 12, 0], [11, 7, 12, 6, 0]],
     [[2, 8, 3, 7, 0], [7, 2, 8, 3, 0]]]
]

if __name__ == "__main__":
    print("Starting training...")
    start_time = time.time()
    game = KnotGame(pd_codes)
    nnet = NNetWrapper(game)
    coach = Coach(game, nnet)
    coach.learn()
    # Save the final model to 'currentModel/current.pth.tar'
    model_path = 'currentModel'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    nnet.save_checkpoint(os.path.join(model_path, 'current.pth.tar'))
    total_time = time.time() - start_time
    print(f"Training completed in {total_time / 60:.2f} minutes.")
    print("Final model saved to currentModel/current.pth.tar.")
