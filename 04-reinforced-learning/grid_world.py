import random

GRID_SIZE = 8
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

START_STATE = (0, 0)
GOAL = (7, 7)
TRAPS = {(1, 3), (2, 3), (3, 3), (4, 3), (5, 5), (6, 5), (3, 6), (4, 6)}

ALPHA = 0.1  # learning rate
GAMMA = 0.9  # discount factor
EPSILON = 0.2  # exploration probability
EPISODES = 5000

# Q-table:
# key: (state, action)
# value: expected future reward
q_table = {}

for row in range(GRID_SIZE):
    for col in range(GRID_SIZE):
        for action in ACTIONS:
            q_table[((row, col), action)] = 0.0


def move(state, action):
    row, col = state

    if action == "UP":
        row -= 1
    elif action == "DOWN":
        row += 1
    elif action == "LEFT":
        col -= 1
    elif action == "RIGHT":
        col += 1

    row = max(0, min(GRID_SIZE - 1, row))
    col = max(0, min(GRID_SIZE - 1, col))

    return row, col


def get_reward(state):
    if state == GOAL:
        return 10
    if state in TRAPS:
        return -10

    return -1


def choose_action(state):
    # Exploration
    if random.random() < EPSILON:
        return random.choice(ACTIONS)

    # Exploitation
    return max(ACTIONS, key=lambda action: q_table[(state, action)])


for episode in range(EPISODES):
    state = START_STATE

    while state != GOAL and state not in TRAPS:
        action = choose_action(state)

        next_state = move(state, action)
        reward = get_reward(next_state)

        best_next_q = max(q_table[(next_state, next_action)] for next_action in ACTIONS)

        old_q = q_table[(state, action)]

        q_table[(state, action)] = old_q + ALPHA * (
            reward + GAMMA * best_next_q - old_q
        )

        state = next_state


def best_action(state):
    return max(ACTIONS, key=lambda action: q_table[(state, action)])


# wiazualizacja polityki
symbols = {"UP": "↑", "DOWN": "↓", "LEFT": "←", "RIGHT": "→"}

print("Learned policy:\n")

for row in range(GRID_SIZE):
    for col in range(GRID_SIZE):
        state = (row, col)

        if state == GOAL:
            print(" ★ ", end="")
        elif state in TRAPS:
            print(" ▓ ", end="")
        else:
            action = best_action(state)
            print(f" {symbols[action]} ", end="")

    print()
