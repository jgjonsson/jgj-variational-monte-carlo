import numpy as np
import matplotlib.pyplot as plt

# Define H0, the one-body term
H0 = np.diag([2, 4, 6, 6, 8, 10])

# Define V, the two-body term (without the -1/2g factor)
V_base = np.array([
    [2, 1, 1, 1, 1, 0],
    [1, 2, 1, 1, 0, 1],
    [1, 1, 2, 0, 1, 1],
    [1, 1, 0, 2, 1, 1],
    [1, 0, 1, 1, 2, 1],
    [0, 1, 1, 1, 1, 2]
])

g_values = np.linspace(-1, 1, 200)
energies = np.zeros((len(g_values), 6))

for i, g in enumerate(g_values):
    V = -0.5 * g * V_base
    H = H0 + V
    eigvals = np.linalg.eigvalsh(H)
    energies[i] = np.sort(eigvals)  # sort for consistent ordering, ground state is first

for state in range(6):
    plt.plot(g_values, energies[:, state], label=f"State {state}")

plt.xlabel("g")
plt.ylabel("Energy")
plt.title("Energy levels vs g")
plt.legend()
plt.grid(True)
plt.show()
