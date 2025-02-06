

import copy
import numpy as np
import snappy
import random 
import config 

class KnotGame:
    def __init__(self, pd_codes, starting_player=1):
        self.pd_codes = pd_codes 
        self.starting_player = starting_player
        self.reset()

    def reset(self):
     
        self.initial_pd_code = copy.deepcopy(random.choice(self.pd_codes))
        
      
        while len(self.initial_pd_code) < config.max_crossings:
        
        
            self.initial_pd_code.append([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        
        self.current_pd_code = copy.deepcopy(self.initial_pd_code)
        self.current_player = self.starting_player
        self.game_over = False
        self.winner = None


    def get_action_size(self):
      
        return config.max_crossings * 2

    def get_valid_moves(self):
      
        action_size = self.get_action_size()
        valids = np.zeros(action_size, dtype=np.float32)
        num_crossings = len(self.current_pd_code)
        for i in range(num_crossings):
            crossing = self.current_pd_code[i]
            if crossing[0][:4] == [0, 0, 0, 0]:
                continue  
            if crossing[0][4] == 0 and crossing[1][4] == 0:
               
                valids[i * 2] = 1  
                valids[i * 2 + 1] = 1  
        return valids

    def make_move(self, action):
        crossing_index = action // 2
        choice_index = action % 2 
        num_crossings = len(self.current_pd_code)
        if crossing_index >= num_crossings:
            raise ValueError("Invalid move: crossing index out of range")
        crossing = self.current_pd_code[crossing_index]
        if crossing[0][:4] == [0, 0, 0, 0]:
            raise ValueError("Invalid move: padding crossing")
        if crossing[0][4] != 0 or crossing[1][4] != 0:
            raise ValueError("Invalid move: crossing already resolved")
       
        crossing[choice_index][4] = 1
        crossing[1 - choice_index][4] = 0  
     
        self.current_player = -self.current_player
      
        if self.is_terminal():
            self.game_over = True
            self.winner = self.get_winner()

    def is_terminal(self):
      
        for crossing in self.current_pd_code:
            if crossing[0][:4] == [0, 0, 0, 0]:
                continue 
            if crossing[0][4] == 0 and crossing[1][4] == 0:
                return False 
        return True 


    def get_winner(self):
     
        final_pd_code = []
        for crossing in self.current_pd_code:
            if crossing[0][:4] == [0, 0, 0, 0]:
                continue  
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
       
        state = []
        for crossing in self.current_pd_code:
            if crossing[0][:4] == [0, 0, 0, 0]:
                state.append(0) 
            elif crossing[0][4] == 1:
                state.append(1) 
            elif crossing[1][4] == 1:
                state.append(2) 
            else:
                state.append(0)  
        
        padding_length = config.max_crossings - len(state)
        if padding_length > 0:
            state.extend([0] * padding_length)
        return np.array(state, dtype=np.int8)

    def get_current_player(self):
        return self.current_player

    def clone(self):
      
        cloned_game = KnotGame(self.pd_codes, self.starting_player)
       
        cloned_game.initial_pd_code = copy.deepcopy(self.initial_pd_code)
        cloned_game.current_pd_code = copy.deepcopy(self.current_pd_code)
        cloned_game.current_player = self.current_player
        cloned_game.game_over = self.game_over
        cloned_game.winner = self.winner
        return cloned_game
