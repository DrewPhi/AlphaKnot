# main.py

from game import KnotGame
from neural_network import NNetWrapper
from coach import Coach
import time
import os
import config  # Import config.py

# Your provided general PD code for knot 6_1
pd_code = [
    [[1, 7, 2, 6, 0], [6, 1, 7, 2, 0]],
    [[3, 10, 4, 11, 0], [10, 4, 11, 3, 0]],
    [[5, 3, 6, 2, 0], [2, 5, 3, 6, 0]],
    [[7, 1, 8, 12, 0], [1, 8, 12, 7, 0]],
    [[9, 4, 10, 5, 0], [4, 10, 5, 9, 0]],
    [[11, 9, 12, 8, 0], [8, 11, 9, 12, 0]]
]

if __name__ == "__main__":
    print("Starting training...")
    start_time = time.time()
    game = KnotGame(pd_code)
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
