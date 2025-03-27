# main.py

from game import KnotGame
from neural_network import NNetWrapper
from coach import Coach
import time
import os
import config

# PD code for your knot
pd_code = [
    [[2, 5, 3, 6, 0], [5, 3, 6, 2, 0]],
    [[4, 10, 5, 9, 0], [9, 4, 10, 5, 0]],
    [[6, 11, 7, 12, 0], [11, 7, 12, 6, 0]],
    [[8, 1, 9, 2, 0], [1, 9, 2, 8, 0]],
    [[10, 4, 11, 3, 0], [3, 10, 4, 11, 0]],
    [[12, 7, 1, 8, 0], [7, 1, 8, 12, 0]]
]

if __name__ == "__main__":
    print("Starting training...")
    start_time = time.time()

    game = KnotGame(pd_code)
    nnet = NNetWrapper(game)
    coach = Coach(game, nnet)
    coach.learn()

    # Save final model
    model_path = 'FinalModel'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    nnet.save_checkpoint(os.path.join(model_path, 'current.pth.tar'))
    total_time = time.time() - start_time
    print(f"Training completed in {total_time / 60:.2f} minutes.")
    print("Final model saved to FinalModel/current.pth.tar.")
