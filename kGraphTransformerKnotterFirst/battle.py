# random_vs_random.py

from knot_graph_game import KnotGraphGame
from arena import Arena
import numpy as np
import torch

def random_player(board, game):
    valids = game.getValidMoves(board, 1).cpu().numpy()
    valid_actions = np.where(valids == 1)[0]
    return np.random.choice(valid_actions)

def main(num_games=1000):
    game = KnotGraphGame()

    player1 = lambda board: random_player(board, game)
    player2 = lambda board: random_player(board, game)

    arena = Arena(player1, player2, game)

    print(f" Running {num_games} random-vs-random games (Player 1 always goes first)...")
    one_won, two_won, draws = arena.playGames(num_games)

    p1_winrate = 100 * one_won / num_games
    p2_winrate = 100 * two_won / num_games
    draw_rate = 100 * draws / num_games

    print("\n Random vs Random Results")
    print(f"Player 1 wins: {one_won} ({p1_winrate:.2f}%)")
    print(f"Player 2 wins: {two_won} ({p2_winrate:.2f}%)")
    print(f"Draws       : {draws} ({draw_rate:.2f}%)")

if __name__ == "__main__":
    main()
