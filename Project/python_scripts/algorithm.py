
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

## Next Function is related to the Q_Learning Task --> currently working on it

def update_Q(prev_state, action, player, reward, current_state, lr, adap_lr, gamma, q_table, update_count_table):
    alpha = lr / (1 + update_count_table[utils.get_ternanry_conversion(prev_state), action] * adap_lr)
    update_count_table[utils.get_ternanry_conversion(prev_state), action] += 1

    if player == 1:
        q_table[prev_state, action] += \
        alpha * (reward + gamma * np.max(q_table[utils.get_ternanry_conversion(current_state)]) - q_table[utils.get_ternanry_conversion(prev_state), action])
    else:
        q_table[prev_state, action] += \
        alpha * (reward + gamma * np.min(q_table[utils.get_ternanry_conversion(current_state)]) - q_table[utils.get_ternanry_conversion(prev_state), action])

    return q_table, update_count_table

def q_learning(map_size, epsilon, gamma, lr, adap_lr, total_reward, state_space, q_table, update_count_table, count_table):

    temp_q_table = q_table.copy()
    temp_update_count = update_count_table.copy()
    temp_count_table = count_table.copy()
    # Initialize starting state and total reward - later used in TD
    current_state = state_space[0]
    
    delta = 0

    for _ in (range(10)):
        
        # Increment visit count for current state
        prev_state = current_state
        temp_count_table[utils.get_ternanry_conversion(prev_state)] += 1
        
        action = utils.epsilon_action(state = current_state, epsilon = epsilon, q_table = temp_q_table)
        
        player = utils.get_player(state = current_state)
        
        # Update State
        current_state = utils.get_next_state(current_state, action, player)
        
        # Get terminating status
        terminate = state_formulation.ongoing_state(map_size, current_state)

        # Get Reward/Payoff of updated state
        reward = state_formulation.get_reward(map_size, current_state)
        total_reward += reward

        # Update q_value for previous state - TD
        old_qsa = temp_q_table[utils.get_ternanry_conversion(prev_state), action]

        temp_q_table, temp_update_count = \
        update_Q(prev_state, action, player, reward, current_state, lr, adap_lr, gamma, q_table, temp_update_count)
        
        delta = np.max([delta, np.abs(old_qsa - temp_q_table[utils.get_ternanry_conversion(prev_state), action])])
        
        if terminate:
            break
    
    # return delta, updated q_table, count_table, update_table, total_reward, 
    return delta, temp_q_table, temp_count_table, temp_update_count, total_reward
