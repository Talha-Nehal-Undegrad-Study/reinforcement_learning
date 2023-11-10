import numpy as np

def get_terminal_states(map_size, total_states):
    terminal_states = []
    for state in total_states:
        # convert state into an array of shape (map_size, map_size)
        split_states = np.array([state[i:i + map_size] for i in range(0, len(state), map_size)])
        
        # check whether the state indicates 'win'
        if np.array([-1] * map_size) in split_states or np.array([-1] * map_size).T in split_states.T or (np.array([-1] * map_size) == np.diag(split_states)).all() or (np.array([-1] * map_size) == np.diag(np.fliplr(split_states))).all():
            terminal_states.append(('win', state))
        
        # check whether the state indicates 'lose'
        elif np.array([1] * map_size) in split_states or np.array([1] * map_size).T in split_states.T or (np.array([1] * map_size) == np.diag(split_states)).all() or (np.array([1] * map_size) == np.diag(np.fliplr(split_states))).all():
            terminal_states.append(('lose', state))
        
        # check whether the state indicates 'draw'
        elif 0 not in state:
            terminal_states.append(('draw', state))

    return terminal_states