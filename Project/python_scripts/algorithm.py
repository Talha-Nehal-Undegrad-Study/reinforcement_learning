
# Importing neccessary scripts and libraries
import numpy as np
from tqdm.auto import tqdm
from python_scripts import state_formulation, utils 

# Function which carries out value iteration. It recieves the grid size, discount factor, whole of state
# space, value_table and q_table both of which will be updated during the value iteration
def value_iteration(map_size, gamma, state_space, value_table, q_table):
    # Get dummy placeholders to update
    temp_value_table = value_table.copy()
    temp_q_table = q_table.copy()
    
    for time in tqdm(range(10)):
        for _, state in enumerate(state_space): # For each state
            if state_formulation.ongoing_state(map_size, state): # check if its ongoing, otherwise continue 
                # in case of terminal state
                actions = utils.get_actions(state) # Get all possible actions
                player = utils.get_player(state) # check which player is to play
                # Get all next possible transition states for each action
                next_states = [utils.get_next_state(state, action, player) for action in actions]
                # Now for each next possible state get its reward using discounted (sum of future_rewards)
                payoffs = [state_formulation.get_reward(map_size, next_state) + ((gamma**time) * temp_value_table[utils.get_ternanry_conversion(next_state)]) for next_state in next_states]
                # Update the payoffs in the q_table --> argmax per row will give policy of each state later
                temp_q_table[utils.get_ternanry_conversion(state), actions] = payoffs
                # If its player 1 i.e. maximizing agent then we want maximization of payoffs else opposite
                # for player 2
                if player == 1:
                    temp_value_table[utils.get_ternanry_conversion(state)] = np.max(payoffs)
                else: 
                    temp_value_table[utils.get_ternanry_conversion(state)] = np.min(payoffs)

    # Note: That whenever we required updating or accessing value/reward of any state, we used its 
    # ternanry hash key for efficient searching.

    # Return the final updated value and q_table.
    return temp_value_table, temp_q_table

## Next Function is related to the Q_Learning Task

def q_learning(map_size, epsilon, gamma, lr, state_space, value_table, q_table):
    
    pass