
import numpy as np
import itertools

def prune_and_get_total_states(grid_size):
    # Our state will be represented as a (grid_size ** 2 x 1) column vector where each element can take either 0, 1, or 2 where 0 represents no move
    # has been made yet on the state/block. 1 represents that the agent who's aiming to maximize the other agent's
    #  reward has made a move on that state while 2 represents the same for the minimizer agent
    
    # Set seed for reproducibility
    np.random.seed(42)

    possible_values = [0, 1, 2]
    # Generate a random array of shape (grid_size ** 2, 1) with values 0, 1, 2, 3 ** 9 such combinations using the powerful iterools library

    all_combinations = np.array(list(itertools.product(possible_values, repeat = grid_size ** 2))) # numpy matrix of shape (3 ** grid_size ** 2, grid_size ** 2)
    
    
    # Now given all the combinations, prune the matrix given the following conditions:
        # 1) The number of 1's should at most be 1 more than the number of 2's in any given combination
        # 2) The number of 2's should at most be 1 less than the number of 1's in any given combination
        
    # Create a dummy list to hold all valid combinations once we come across them
    pruned_states = []
    for row in range(all_combinations.shape[0]):
        combination = all_combinations[row, :]
        
        one_occurences = np.sum(combination == 1)
        two_occurences = np.sum(combination == 2)

        if one_occurences - two_occurences == 1 or one_occurences == two_occurences:
            pruned_states.append(list(combination))
            
    return pruned_states

# Function which evaluates whether a state is ongoing i.e. not terminated yet
def ongoing_state(map_size, state):
    # Take the state and convert it into a matrix form for easier manipulation
    split_states = [state[i:i + map_size] for i in range(0, len(state), map_size)]

    # We now write masks for win and lose conditions i.e. we will be looking out for same elements across
    # rows, cols, main and off diagnol. 

    win_conditions = np.sum([[1] * map_size in split_states, 
                             [1] * map_size in list(map(list, zip(*split_states))), 
                             ([1] * map_size == np.diag(split_states)).all(), 
                             ([1] * map_size == np.diag(np.fliplr(split_states))).all()])
    
    lose_conditions = np.sum([[2] * map_size in split_states, 
                              [2] * map_size in list(map(list, zip(*split_states))), 
                              ([2] * map_size == np.diag(split_states)).all(), 
                              ([2] * map_size == np.diag(np.fliplr(split_states))).all()])
    
    # If win conditions and lose conditions are > 0 then we know that its for sure not an ongoing state
    # or if state contains no zero then its not an ongoing state
    if win_conditions + lose_conditions > 0 or 0 not in state:
        return False
    
    # Otherwise return true

    return True

# Function which evaluates the reward of a state. 1 if win, -1 if lose, 0 otherwise
def get_reward(map_size, state):
    
    # Similar to as the ongoing function, we use the win and lose masks
    split_states = [state[i:i + map_size] for i in range(0, len(state), map_size)]

    win_conditions = np.sum([[1] * map_size in split_states, 
                             [1] * map_size in list(map(list, zip(*split_states))), 
                             ([1] * map_size == np.diag(split_states)).all(), 
                             ([1] * map_size == np.diag(np.fliplr(split_states))).all()])
    
    lose_conditions = np.sum([[2] * map_size in split_states, 
                              [2] * map_size in list(map(list, zip(*split_states))), 
                              ([2] * map_size == np.diag(split_states)).all(), 
                              ([2] * map_size == np.diag(np.fliplr(split_states))).all()])
    
    # Note that win_conditions and lose conditions both can be indiviually greater than 0 but not together
    # Therefore the following also acts as a prune condition which negates such states 
    
    return 1 if win_conditions > 0 and lose_conditions == 0 else -1 if lose_conditions > 0 and win_conditions == 0 else 0
                                    
    
                                        
