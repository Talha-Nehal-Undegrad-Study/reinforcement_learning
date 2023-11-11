import numpy as np

def get_terminal_states(map_size, total_states):
    terminal_states = []

    for state in total_states:
        # Convert state into an array of shape (map_size, map_size)
        matrix = np.array([state[i: i + map_size] for i in range(0, len(state), map_size)])

        draw = True
        count_lose = 0
        count_win = 0

        for i in range(matrix.shape[0]):
            # Check for winning and losing conditions
            mask_lose = np.all(np.unique(matrix[i, :]) == 1) or \
                        np.all(np.unique(matrix[:, i]) == 1) or \
                        (i == 0 and np.all(np.diag(matrix) == 1)) or \
                        (i == 0 and np.all(np.diag(np.fliplr(matrix)) == 1))

            mask_win = np.all(np.unique(matrix[i, :]) == -1) or \
                       np.all(np.unique(matrix[:, i]) == -1) or \
                       (i == 0 and np.all(np.diag(matrix) == -1)) or \
                       (i == 0 and np.all(np.diag(np.fliplr(matrix)) == -1))

            if mask_win:
                count_win += 1
                draw = False

            if mask_lose:
                count_lose += 1
                draw = False

        # Check the terminal state classification
        if draw:
            if np.sum(matrix == 0) == 0:
                terminal_states.append(('Draw', state))
            else:
                terminal_states.append(('Ongoing', state))
        # Note: count_win or count_lose indiually can be greater > 1 but not together i.e. if both count_win and count_lose are non-zero then that's invalid state
        elif count_lose != 0 and count_win != 0:
            terminal_states.append(('Invalid', state))
        elif count_win >= 1:
            terminal_states.append(('Win', state))
        elif count_lose >= 1:
            terminal_states.append(('Lose', state))

    return terminal_states


def final_prune(labeled_states, total_states):
    terminal_states = [element for element in labeled_states if element[0] != 'Ongoing']
    invalid_states = [element[1] for element in terminal_states if element[0] == 'Invalid']
    total_states = [lst for lst in total_states if lst not in invalid_states]
    terminal_states = [lst for lst in terminal_states if lst[1] not in invalid_states]
    return total_states, terminal_states