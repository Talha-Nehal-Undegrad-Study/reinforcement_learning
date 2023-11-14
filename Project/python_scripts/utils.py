import numpy as np

# Python script which contains helper functions for value iteration

# A function which gives which player has to play given a state. If the number of 1's and 2's are equal
# then its player 1's turn. If number of 1's are greater than number of 2's then its player's 2 turn.
def get_player(state):
    values, counts = np.unique(state, return_counts = True)
    if len(values) == 1: return 1
    if len(values) == 2: return 2
    if len(values) == 3: return 2 if counts[1] > counts[2] else 1

# A function which given a state and action (i.e. from 0 - map_grid_size ** 2 - 1) and player (1 or 2)
# updates the state at the index of interest. Note: We create a copy of the state so as to avoid changes 
# in the original state
def get_next_state(state, action, player):
    new_state = state.copy()
    new_state[action] = player
    return new_state

# A function which when given a state checks all the indices where the state is 0 and then those indices 
# becomes all possible actions. 
def get_actions(state):
    return [index for index, value in enumerate(state) if value == 0]

# A very useful and important function used to hash the states. This way we can index each state, reward,
# vectors by not the state but rather through a unique idnetifier which in this case is the tennary
# decimal conversion.
def get_ternanry_conversion(state):
    return int(''.join([str(i) for i in state]), base = 3)
