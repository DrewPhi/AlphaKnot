# game.py

import config
import copy
import numpy as np
import snappy

class KnotGame:
    def __init__(self, pd_code, starting_player=1):
        self.initial_pd_code = pd_code
        self.starting_player = starting_player
        self.reset()

    def reset(self):
        # Initialize the game state
        self.current_pd_code = copy.deepcopy(self.initial_pd_code)
        self.current_player = self.starting_player
        self.game_over = False
        self.winner = None

    def get_action_size(self):
        # Total number of actions: number of crossings * 2
        return len(self.initial_pd_code) * 2

    def get_valid_moves(self):
        # Return a numpy array of valid moves (1 for valid, 0 for invalid)
        action_size = self.get_action_size()
        valids = np.zeros(action_size, dtype=np.float32)
        for i, crossing in enumerate(self.current_pd_code):
            # crossing[x][4] is the 5th element (0 or 1) indicating if resolved
            if crossing[0][4] == 0 and crossing[1][4] == 0:
                valids[i * 2] = 1   # First choice
                valids[i * 2 + 1] = 1  # Second choice
        return valids

    def make_move(self, action):
        # Resolve a crossing with the specified choice
        crossing_index = action // 2
        choice_index = action % 2
        crossing = self.current_pd_code[crossing_index]

        if crossing[0][4] != 0 or crossing[1][4] != 0:
            raise ValueError("Invalid move: crossing already resolved")

        crossing[choice_index][4] = 1
        crossing[1 - choice_index][4] = 0

        # Switch players
        self.current_player = -self.current_player

        # Check if the game is over
        if self.is_terminal():
            self.game_over = True
            self.winner = self.get_winner()

    def is_terminal(self):
        # The game is over when all crossings are resolved
        return all(c[0][4] == 1 or c[1][4] == 1 for c in self.current_pd_code)

    def get_winner(self):
        # Build the final PD code from chosen resolutions
        final_pd_code = []
        for crossing in self.current_pd_code:
            if crossing[0][4] == 1:
                final_pd_code.append(crossing[0][:4])
            elif crossing[1][4] == 1:
                final_pd_code.append(crossing[1][:4])

        is_unknot = self.is_unknot(final_pd_code)

        # If knotter_first=True, Player 1 is the knotter => wins if not is_unknot
        # If knotter_first=False, Player 1 is the unknotter => wins if is_unknot
        if config.knotter_first:
            return 1 if not is_unknot else -1
        else:
            return 1 if is_unknot else -1

    def is_unknot(self, pd_code):
        link = snappy.Link(pd_code)
        alex = link.alexander_polynomial()
        if alex != 1:
            return False
        jones = link.jones_polynomial()
        return jones == 1

    def get_canonical_form(self):
        # Return an array encoding all crossing states plus current_player
        state = []
        for crossing in self.current_pd_code:
            if crossing[0][4] == 1:
                state.append(1)
            elif crossing[1][4] == 1:
                state.append(2)
            else:
                state.append(0)
        state.append(self.current_player)
        return np.array(state, dtype=np.int8)

    def get_current_player(self):
        return self.current_player

    def clone(self):
        cloned_game = KnotGame(self.initial_pd_code, self.starting_player)
        cloned_game.current_pd_code = copy.deepcopy(self.current_pd_code)
        cloned_game.current_player = self.current_player
        cloned_game.game_over = self.game_over
        cloned_game.winner = self.winner
        return cloned_game
