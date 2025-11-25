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
energies = []

for g in g_values:
    V = -0.5 * g * V_base
    H = H0 + V
    eigvals = np.linalg.eigvalsh(H)
    energies.append(np.min(eigvals))  # ground state energy

plt.plot(g_values, energies)
plt.xlabel("g")
plt.ylabel("Ground state energy")
plt.title("Ground state energy vs g")
plt.grid(True)
plt.show()
