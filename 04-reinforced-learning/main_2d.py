import random

import matplotlib.pyplot as plt
import numpy as np

# Parametry środowiska
start = [0, 0]
goal = [10, 10]
obstacles = [[3, 3], [5, 5], [7, 2], [6, 6]]


# Strategia: losowy ruch
def policy(state):
    return random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])


# Symulacja
state = start[:]
path = [state[:]]
hits = 0

for step in range(200):
    action = policy(state)
    next_state = [state[0] + action[0], state[1] + action[1]]

    if next_state in obstacles:
        hits += 1
        continue  # nie ruszaj się w przeszkodę

    state = next_state
    path.append(state[:])

    if state == goal:
        print(f"Dotarł do celu w {step+1} krokach.")
        break

# Przygotuj dane do wykresu
x_vals = [p[0] for p in path]
y_vals = [p[1] for p in path]
obs_x = [o[0] for o in obstacles]
obs_y = [o[1] for o in obstacles]

# Rysowanie
plt.figure(figsize=(8, 8))
plt.plot(x_vals, y_vals, marker="o", label="Droga agenta")
plt.scatter(obs_x, obs_y, color="red", label="Przeszkody", s=100)
plt.scatter(*goal, color="green", label="Cel", s=100)
plt.scatter(*start, color="blue", label="Start", s=100)
plt.title("Ruch agenta w środowisku 2D")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.legend()
plt.axis("equal")
plt.show()

print(f"Liczba trafień w przeszkody: {hits}")
