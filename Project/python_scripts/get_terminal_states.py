import numpy as np

def get_terminal_states(map_size, total_states):
    terminal_states = [('draw', []), ('lose', []), ('win', [])]
    for state in total_states:
        # convert state into an array of shape (map_size, map_size)
        split_states = np.array([state[i:i + map_size] for i in range(0, len(state), map_size)])
        
        # check whether the state indicates 'win'
        if np.array([-1] * map_size) in split_states or np.array([-1] * map_size).T in split_states.T or np.array([-1] * map_size).T == np.diag(split_states):
            terminal_states[2][1].append(state)
        
        # check whether the state indicates 'lose'
        elif np.array([1] * map_size) in split_states or np.array([1] * map_size).T in split_states.T or np.array([1] * map_size).T == np.diag(split_states):
            terminal_states[1][1].append(state)
        
        # check whether the state indicates 'draw'
        elif 0 not in state:
            terminal_states[0][1].append(state)

    return terminal_states