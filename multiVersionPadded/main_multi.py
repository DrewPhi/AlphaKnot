# main_multi.py

from game_multi import KnotGame
from neural_network_multi import NNetWrapper
from coach_multi import Coach
import time
import os
import config  # Import config.py

# Your provided general PD codes for multiple knots
pd_codes = [
    [[[1, 5, 2, 4,0], [4, 1, 5, 2,0]], [[3, 1, 4, 6,0], [1, 4, 6, 3,0]], [[5, 3, 6, 2,0], [2, 5, 3, 6,0]]], #3_1
    [[[4, 2, 5, 1,0], [1, 4, 2, 5,0]], [[8, 6, 1, 5,0], [5, 8, 6, 1,0]], [[6, 3, 7, 4,0], [3, 7, 4, 6,0]], [[2, 7, 3, 8,0], [7, 3, 8, 2,0]]], #4_1
    [[[2, 8, 3, 7,0], [7, 2, 8, 3,0]], [[4, 10, 5, 9,0], [9, 4, 10, 5,0]], [[6, 2, 7, 1,0], [1, 6, 2, 7,0]], [[8, 4, 9, 3,0], [3, 8, 4, 9,0]], [[10, 6, 1, 5,0], [5, 10, 6, 1,0]]],#5_1
    [
    [[1, 7, 2, 6, 0], [6, 1, 7, 2, 0]],
    [[3, 10, 4, 11, 0], [10, 4, 11, 3, 0]],
    [[5, 3, 6, 2, 0], [2, 5, 3, 6, 0]],
    [[7, 1, 8, 12, 0], [1, 8, 12, 7, 0]],
    [[9, 4, 10, 5, 0], [4, 10, 5, 9, 0]],
    [[11, 9, 12, 8, 0], [8, 11, 9, 12, 0]]
], #6_1
[
    [[2, 10, 3, 9, 0], [9, 2, 10, 3, 0]],
    [[4, 14, 5, 13, 0], [13, 4, 14, 5, 0]],
    [[6, 12, 7, 11, 0], [11, 6, 12, 7, 0]],
    [[8, 2, 9, 1, 0], [1, 8, 2, 9, 0]],
    [[10, 8, 11, 7, 0], [7, 10, 8, 11, 0]],
    [[12, 6, 13, 5, 0], [5, 12, 6, 13, 0]],
    [[14, 4, 1, 3, 0], [3, 14, 4, 1, 0]]
]
,#7_2
[
    [[1, 9, 2, 8, 0], [8, 1, 9, 2, 0]],
    [[3, 7, 4, 6, 0], [6, 3, 7, 4, 0]],
    [[5, 12, 6, 13, 0], [12, 6, 13, 5, 0]],
    [[7, 3, 8, 2, 0], [2, 7, 3, 8, 0]],
    [[9, 1, 10, 16, 0], [1, 10, 16, 9, 0]],
    [[11, 15, 12, 14, 0], [14, 11, 15, 12, 0]],
    [[13, 4, 14, 5, 0], [4, 14, 5, 13, 0]],
    [[15, 11, 16, 10, 0], [10, 15, 11, 16, 0]]
]
#8_1

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
