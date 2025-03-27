import os
import numpy as np
from game import KnotGame
from neural_network import NNetWrapper
import random

Knotter_first = False

def play_game(nnet, starting_player, ai_player):
    pd_code = [
    [[2, 5, 3, 6, 0], [5, 3, 6, 2, 0]],
    [[4, 10, 5, 9, 0], [9, 4, 10, 5, 0]],
    [[6, 11, 7, 12, 0], [11, 7, 12, 6, 0]],
    [[8, 1, 9, 2, 0], [1, 9, 2, 8, 0]],
    [[10, 4, 11, 3, 0], [3, 10, 4, 11, 0]],
    [[12, 7, 1, 8, 0], [7, 1, 8, 12, 0]]
]
    game = KnotGame(pd_code, starting_player)
    while not game.game_over:
        if game.get_current_player() == ai_player:
            state = game.get_canonical_form()
            pi, _ = nnet.predict(state)
            valids = game.get_valid_moves()
            pi = pi * valids
            if np.sum(pi) > 0:
                pi /= np.sum(pi)
            else:
                pi = valids / np.sum(valids)
            action = np.argmax(pi)
        else:
            valids = game.get_valid_moves()
            valid_actions = np.where(valids == 1)[0]
            action = random.choice(valid_actions)
        game.make_move(action)
    return game.winner

def main():
    model_path = 'Knotter_Not_First/Knotter_Not_First_best.pth.tar'

    pd_code = [
        [[2, 5, 3, 6,0], [5, 3, 6, 2,0]],
        [[4, 10, 5, 9,0],[9, 4, 10, 5,0]],
        [[6, 11, 7, 12,0],[11, 7, 12, 6,0]],
        [[8, 1, 9, 2,0], [1, 9, 2, 8,0]],
        [[10,4, 11,3,0], [3, 10,4,11,0]],
        [[12,7, 1, 8,0],[7, 1, 8,12,0]]
    ]
    game = KnotGame(pd_code)
    nnet = NNetWrapper(game)
    nnet.load_checkpoint(model_path)

    ai_player = 1 if Knotter_first else -1
    random_player = -ai_player

    ai_first_wins = 0
    ai_second_wins = 0

    n_games = 200

    print(f"--- Evaluating AI as Player {ai_player} ---")

    # AI goes first
    for i in range(n_games):
        winner = play_game(nnet, ai_player, ai_player)
        if winner == ai_player:
            ai_first_wins += 1
        print(f"[AI First] Game {i+1}/{n_games}: Winner = {winner}")

    # AI goes second
    for i in range(n_games):
        winner = play_game(nnet, random_player, ai_player)
        if winner == ai_player:
            ai_second_wins += 1
        print(f"[AI Second] Game {i+1}/{n_games}: Winner = {winner}")

    print("\n=== Evaluation Complete ===")
    print(f"AI won {ai_first_wins}/200 games when going FIRST ({ai_first_wins / 2:.1f}%).")
    print(f"AI won {ai_second_wins}/200 games when going SECOND ({ai_second_wins / 2:.1f}%).")

if __name__ == "__main__":
    main()
