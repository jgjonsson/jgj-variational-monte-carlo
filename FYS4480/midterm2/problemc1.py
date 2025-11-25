import numpy as np
import matplotlib.pyplot as plt

# Use only the first 5 states
H0_reduced = np.diag([2, 4, 6, 6, 8])  # first 5 diagonal elements
V_base_reduced = np.array([
    [2, 1, 1, 1, 1],
    [1, 2, 1, 1, 0],
    [1, 1, 2, 0, 1],
    [1, 1, 0, 2, 1],
    [1, 0, 1, 1, 2]
])

g_values = np.linspace(-1, 1, 200)
energies_reduced = np.zeros((len(g_values), 5))

for i, g in enumerate(g_values):
    V = -0.5 * g * V_base_reduced
    H = H0_reduced + V
    eigvals = np.linalg.eigvalsh(H)
    energies_reduced[i] = np.sort(eigvals)

# Plot only the ground state (state 0)
plt.plot(g_values, energies_reduced[:, 0], label="Ground State")

plt.xlabel("g")
plt.ylabel("Energy")
plt.title("Ground state energy vs g (2p-2h only)")
plt.legend()
plt.grid(True)
plt.show()
