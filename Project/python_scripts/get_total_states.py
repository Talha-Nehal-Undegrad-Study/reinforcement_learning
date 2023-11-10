
import numpy as np
import itertools

def prune_and_get_total_states(grid_size):
    # Our state will be represented as a (grid_size ** 2 x 1) column vector where each element can take either 0, 1, or -1 where 0 represents no move
    # has been made yet on the state/block. 1 represents that the agent who's aiming to maximize (denoted by M_m) the other agent's
    #  reward has made a move on that state while -1 represents the same for the minimizer agent (denoted by M_a)
    

    # Set seed for reproducibility
    np.random.seed(42)

    possible_values = [0, 1, -1]
    # Generate a random array of shape (grid_size ** 2, 1) with values 0, 1, -1, 3 ** 9 such combinations using the powerful iterools library

    all_combinations = np.array(list(itertools.product(possible_values, repeat = grid_size ** 2))) # numpy matrix of shape (3 ** grid_size ** 2, grid_size ** 2)
    
    
    # Now given all the combinations, prune the matrix given the following conditions:
        # 1) The number of 1's should at most be 1 more than the number of -1's in any given combination
        # 2) The number of -1's should at most be 1 less than the number of 1's in any given combination
        
    # Create a dummy list to hold all valid combinations once we come across them
    pruned_states = []
    for row in range(all_combinations.shape[0]):
        combination = all_combinations[row, :]
        
        one_occurences = np.sum(combination == 1)
        minus_one_occurences = np.sum(combination == -1)

        if minus_one_occurences - one_occurences == 1 or one_occurences == minus_one_occurences:
            pruned_states.append(list(combination))

    # Return the list where at each list we have a possible state
    # pruned_states = [element[0] for lst in pruned_states for element in lst]

    return pruned_states

                                    
    
                                        
