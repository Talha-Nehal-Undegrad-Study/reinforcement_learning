import numpy as np

def get_terminal_states(map_size, total_states):
    terminal_states = []
    
    for state in total_states:
        # convert state into an array of shape (map_size, map_size)
        matrix = np.array([state[i:i + map_size] for i in range(0, len(state), map_size)])

        draw = True
        lose = False
        win = False
        for i in range(matrix.shape[0]):
            if (np.all(np.unique(matrix[i, :]) == 1) and matrix[i, 0] == 1) or \
               (np.all(np.unique(matrix[:, i]) == 1) and matrix[0, i] == 1) or \
               (len(set(np.diag(matrix))) == 1 and matrix[i, i] == 1) or \
               (len(set(np.diag(np.fliplr(matrix)))) == 1 and matrix[i, -1 - i] == 1):
                terminal_states.append(('Lose', state))
                draw = False
                lose = True
            if (np.all(np.unique(matrix[i, :]) == -1) and matrix[i, 0] == -1) or \
                 (np.all(np.unique(matrix[:, i]) == -1) and matrix[0, i] == -1) or \
                 (len(set(np.diag(matrix))) == 1 and matrix[i, i] == -1) or \
                 (len(set(np.diag(np.fliplr(matrix)))) == 1 and matrix[i, -1 - i] == -1):
                terminal_states.append(('Win', state))
                draw = False
                win = True
            if win and lose:
                terminal_states.append(('Invalid', state))
            elif i == matrix.shape[0] - 1 and draw:
                if np.sum(matrix == 0) == 0:
                    terminal_states.append(('Draw', state))
                else:
                    terminal_states.append(('Ongoing', state))

    return terminal_states

def final_prune(labeled_states, total_states):
    terminal_states = [element for element in labeled_states if element[0] != 'Ongoing']
    invalid_states = [element[1] for element in terminal_states if element[0] == 'Invalid']
    total_states = [lst for lst in total_states if lst not in invalid_states]
    terminal_states = [lst for lst in terminal_states if lst[1] not in invalid_states]
    return total_states, terminal_states