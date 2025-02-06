# game_multi.py

import copy
import numpy as np
import snappy
import random  # Import random module
import config  # Import config.py

class KnotGame:
    def __init__(self, pd_codes, starting_player=1):
        self.pd_codes = pd_codes  # List of PD codes
        self.starting_player = starting_player
        self.reset()

    def reset(self):
        # Randomly select a PD code for this game
        self.initial_pd_code = copy.deepcopy(random.choice(self.pd_codes))
        
        # Pad self.initial_pd_code to ensure it has exactly config.max_crossings entries
        while len(self.initial_pd_code) < config.max_crossings:
            # Append a padding crossing with no real data
            # Both "resolutions" are zeros, indicating no actual crossing
            self.initial_pd_code.append([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        
        self.current_pd_code = copy.deepcopy(self.initial_pd_code)
        self.current_player = self.starting_player
        self.game_over = False
        self.winner = None


    def get_action_size(self):
        # Total number of actions: max_crossings times 2 choices per crossing
        return config.max_crossings * 2

    def get_valid_moves(self):
        # Return a numpy array of valid moves (1 for valid, 0 for invalid)
        action_size = self.get_action_size()
        valids = np.zeros(action_size, dtype=np.float32)
        num_crossings = len(self.current_pd_code)
        for i in range(num_crossings):
            crossing = self.current_pd_code[i]
            if crossing[0][:4] == [0, 0, 0, 0]:
                continue  # Skip padding crossings
            if crossing[0][4] == 0 and crossing[1][4] == 0:
                # Both choices are valid
                valids[i * 2] = 1  # First choice
                valids[i * 2 + 1] = 1  # Second choice
        return valids

    def make_move(self, action):
        crossing_index = action // 2
        choice_index = action % 2  # 0 or 1
        num_crossings = len(self.current_pd_code)
        if crossing_index >= num_crossings:
            raise ValueError("Invalid move: crossing index out of range")
        crossing = self.current_pd_code[crossing_index]
        if crossing[0][:4] == [0, 0, 0, 0]:
            raise ValueError("Invalid move: padding crossing")
        if crossing[0][4] != 0 or crossing[1][4] != 0:
            raise ValueError("Invalid move: crossing already resolved")
        # Set the fifth element to 1 for the chosen resolution
        crossing[choice_index][4] = 1
        crossing[1 - choice_index][4] = 0  # Ensure the alternative is not chosen
        # Switch players
        self.current_player = -self.current_player
        # Check if the game is over
        if self.is_terminal():
            self.game_over = True
            self.winner = self.get_winner()

    def is_terminal(self):
        # The game is over when all real crossings are resolved
        for crossing in self.current_pd_code:
            if crossing[0][:4] == [0, 0, 0, 0]:
                continue  # Skip padding crossings
            if crossing[0][4] == 0 and crossing[1][4] == 0:
                return False  # Found an unresolved crossing
        return True  # All real crossings are resolved


    def get_winner(self):
        # Build the final PD code
        final_pd_code = []
        for crossing in self.current_pd_code:
            if crossing[0][:4] == [0, 0, 0, 0]:
                continue  # Skip padding crossings
            if crossing[0][4] == 1:
                final_pd_code.append(crossing[0][:4])
            elif crossing[1][4] == 1:
                final_pd_code.append(crossing[1][:4])
        is_unknot = self.is_unknot(final_pd_code)
        return 1 if is_unknot else -1


    def is_unknot(self, pd_code):
        link = snappy.Link(pd_code)
        alex = link.alexander_polynomial()
        if alex != 1:
            return False
        jones = link.jones_polynomial()
        return jones == 1

    def get_canonical_form(self):
        # Return a fixed-size representation of the game state
        state = []
        for crossing in self.current_pd_code:
            if crossing[0][:4] == [0, 0, 0, 0]:
                state.append(0)  # Use 0 for padding crossings
            elif crossing[0][4] == 1:
                state.append(1)  # First choice selected
            elif crossing[1][4] == 1:
                state.append(2)  # Second choice selected
            else:
                state.append(0)  # Unresolved
        # Pad the state vector to max_crossings
        padding_length = config.max_crossings - len(state)
        if padding_length > 0:
            state.extend([0] * padding_length)
        return np.array(state, dtype=np.int8)

    def get_current_player(self):
        return self.current_player

    def clone(self):
        # Clone the current game without re-randomizing the PD code
        cloned_game = KnotGame(self.pd_codes, self.starting_player)
        # Overwrite fields so that clone matches the current game state exactly
        cloned_game.initial_pd_code = copy.deepcopy(self.initial_pd_code)
        cloned_game.current_pd_code = copy.deepcopy(self.current_pd_code)
        cloned_game.current_player = self.current_player
        cloned_game.game_over = self.game_over
        cloned_game.winner = self.winner
        return cloned_game
