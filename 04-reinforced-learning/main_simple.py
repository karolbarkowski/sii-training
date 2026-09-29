import matplotlib.pyplot as plt
import numpy as np

# Definicja prostego środowiska
# Tworzymy bardzo uproszczony świat: agent zaczyna na pozycji 0 i ma za zadanie dotrzeć do pozycji 10.

# 📖 Wytłumaczenie:
# Świat = jedna linia liczb od 0 do 10.
# Agent w każdym kroku może:
# przesunąć się o +1 (do przodu)
# lub o -1 (wstecz, ale tego nie chce!)
# 💡 Ćwiczenie: Zmodyfikuj goal_position, aby agent musiał dotrzeć np. do 5 lub 15.
goal_position = 10
state = 0


# Definicja strategii (polityki)
def policy(state):
    if np.random.rand() < 0.9:
        return 1  # krok do przodu
    else:
        return -1  # krok wstecz


# Symulacja ruchu agenta
path = [state]

while state != goal_position:
    action = policy(state)
    state += action
    path.append(state)

# Wizualizacja drogi agenta
plt.plot(path, marker="o")
plt.title("Droga agenta do celu")
plt.xlabel("Krok")
plt.ylabel("Pozycja")
plt.grid()
plt.show()
